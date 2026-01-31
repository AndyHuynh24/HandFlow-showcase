"""
HandFlow MacroPad Detector
==========================

8-marker corner-based detection system for paper macro pad.

Marker Layout (as seen by camera):
    [TL: Set ID]              [TR: ID 4]
           |                       |
    [ML: ID 5]                [MR: ID 6]
           |                       |
    [BL2: ID 9]              [BR2: ID 10]  <- Outer fallback markers
           |                       |
           [BL: ID 7]    [BR: ID 8]        <- Inner bottom markers

Detection region corners:
- Top-left: TL marker's top-right corner
- Top-right: TR marker's top-left corner
- Bottom-left: BL's top-left corner OR BL2's bottom-right corner (fallback)
- Bottom-right: BR's top-right corner OR BR2's bottom-left corner (fallback)

Divides into 4x2 grid for 8 buttons.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class MacroPadDetection:
    """Detection result for a macro pad."""
    set_marker_id: int  # The top-left marker ID that identifies this set
    detection_region: np.ndarray  # 4 corners of the detection region (TL, TR, BR, BL)
    grid_cells: List[np.ndarray]  # 8 grid cell polygons (index 0-7)
    marker_positions: Dict[int, np.ndarray]  # Detected/estimated marker centers
    estimated_markers: set  # Set of marker IDs that were estimated


class MacroPadDetector:
    """
    Detects macro pad using 8 markers with fallback bottom corners.

    Logic mirrors ArUcoScreenDetector:
    - Uses TL, TR, BL, BR as primary corner markers
    - BL2, BR2 provide fallback bottom corners when BL/BR are occluded
    - Requires at least 3 visible corners to estimate the 4th
    - Uses Homography (4 pts) or Affine (3 pts) transform for estimation
    - Middle markers (ML, MR) help with corner estimation

    Marker Layout:
        [TL]              [TR]
           |              |
        [ML]              [MR]
           |              |
        [BL2]            [BR2]  <- Fallback outer markers
           |              |
           [BL]      [BR]       <- Primary inner markers
    """
    
    # Position names for debugging and mapping
    POS_TL = 'TL'
    POS_TR = 'TR'
    POS_BL = 'BL'
    POS_BR = 'BR'

    # Middle markers 
    POS_ML = 'ML'
    POS_MR = 'MR'

    # Fallback bottom markers (directly under ML/MR)
    POS_BL2 = 'BL2'
    POS_BR2 = 'BR2'

    # Map constant IDs
    # TL is set-dependent
    FIXED_IDS = {
        POS_TR: 4,
        POS_ML: 5,
        POS_MR: 6,
        POS_BL: 7,
        POS_BR: 8,
        POS_BL2: 9,   # Fallback for BL
        POS_BR2: 10,  # Fallback for BR
    }

    # Grid layout: 4 columns x 2 rows = 8 buttons
    GRID_COLS = 4
    GRID_ROWS = 2
    
    # Detection mode presets
    MODE_BALANCED = "balanced"
    MODE_MOTION_PRIORITY = "motion_priority"

    def __init__(self, aruco_dict: int = cv2.aruco.DICT_4X4_50, detection_mode: str = "balanced"):
        """
        Initialize MacroPad detector.

        Args:
            aruco_dict: ArUco dictionary type
            detection_mode: Detection preset
                - "balanced": Good motion blur tolerance with stable static detection
                - "motion_priority": Maximum motion blur tolerance, may jitter when static
        """
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self._detection_mode = detection_mode
        detector_params = cv2.aruco.DetectorParameters()

        # === Common settings ===
        detector_params.adaptiveThreshWinSizeMin = 7
        detector_params.adaptiveThreshWinSizeStep = 5
        detector_params.minCornerDistanceRate = 0.02
        detector_params.minMarkerDistanceRate = 0.02
        detector_params.errorCorrectionRate = 0.6  # Maximum error tolerance

        if detection_mode == self.MODE_MOTION_PRIORITY:
            # === MOTION PRIORITY MODE ===
            # Aggressive settings for maximum motion blur tolerance
            # Trade-off: More jitter/wobble when static, possible false positives

            detector_params.adaptiveThreshWinSizeMax = 35

            detector_params.polygonalApproxAccuracyRate = 0.12  # Very lenient (default 0.03)

            # No corner refinement - refinement fails on blurry corners
            detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE

            # Very lenient bit extraction - ignore more of the blurry edges
            detector_params.perspectiveRemovePixelPerCell = 5
            detector_params.perspectiveRemoveIgnoredMarginPerCell = 0.28

            # More lenient marker size filtering
            detector_params.minMarkerPerimeterRate = 0.02  # Allow smaller markers

        else:
            # === BALANCED MODE (default) ===
            # Good motion blur tolerance with reasonable static stability
            detector_params.adaptiveThreshWinSizeMax = 28

            # Moderately lenient contour approximation
            detector_params.polygonalApproxAccuracyRate = 0.08  # (default 0.03)

            # CONTOUR refinement
            detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
            detector_params.cornerRefinementWinSize = 7
            detector_params.cornerRefinementMaxIterations = 20

            # Moderately lenient bit extraction
            detector_params.perspectiveRemovePixelPerCell = 6
            detector_params.perspectiveRemoveIgnoredMarginPerCell = 0.22

        self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, detector_params)
        
        # Detection state
        self._last_detection: Optional[MacroPadDetection] = None
        self._detection_valid = False
        self._current_set_marker_id: Optional[int] = None
        
        # Marker cache for occlusion handling
        # Stores (center, corners) for the 4 CORNER markers
        self._marker_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {} 
        self._estimated_pos_names: set = set()
        
    def detect(self, frame: np.ndarray, known_set_ids: List[int] = None) -> bool:
        """
        Detect macro pad markers.
        Requires at least 3 of the 4 CORNER markers to be visible.
        """
        if known_set_ids is None:
            known_set_ids = [12, 13, 14]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids = None, None

        # -------------------------------------------------
        # Attempt 1: Standard grayscale
        # -------------------------------------------------
        corners, ids, _ = self._detector.detectMarkers(gray)

        # -------------------------------------------------
        # Attempt 2: If not enough markers, try with CLAHE (contrast enhancement)
        # -------------------------------------------------
        if ids is None or len(ids) < 3:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            corners2, ids2, _ = self._detector.detectMarkers(enhanced)
            if ids2 is not None and (ids is None or len(ids2) > len(ids)):
                corners, ids = corners2, ids2

        # -------------------------------------------------
        # Attempt 3: If still not enough, try with slight blur (reduces noise)
        # -------------------------------------------------
        if ids is None or len(ids) < 3:
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            corners3, ids3, _ = self._detector.detectMarkers(blurred)
            if ids3 is not None and (ids is None or len(ids3) > len(ids)):
                corners, ids = corners3, ids3
        
        if ids is None:
            self._detection_valid = False
            return False
        
        # -------------------------------------------------
        # 1. Map all detected markers by ID
        # -------------------------------------------------
        detected_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {} # id -> (center, corners)
        for i, marker_id in enumerate(ids.flatten()):
            marker_corners = corners[i][0]
            center = np.mean(marker_corners, axis=0)
            detected_data[marker_id] = (center, marker_corners)
            
        # -------------------------------------------------
        # 2. Identify Set ID (TL marker)
        # -------------------------------------------------
        # Find which set marker is currently visible
        set_marker_id = None
        for sid in known_set_ids:
            if sid in detected_data:
                set_marker_id = sid
                break

        # Check if switching to a different set
        if set_marker_id is not None and self._current_set_marker_id is not None:
            if set_marker_id != self._current_set_marker_id:
                # Switching sets - clear cache since marker positions are completely different
                self._marker_cache.clear()
                self._estimated_pos_names.clear()

        # Use cached set ID only if no set marker detected at all
        if set_marker_id is None and self._current_set_marker_id is not None:
            set_marker_id = self._current_set_marker_id

        if set_marker_id is None:
            self._detection_valid = False
            return False

        self._current_set_marker_id = set_marker_id
        
        # -------------------------------------------------
        # 3. Map positions to IDs for this frame
        # -------------------------------------------------
        pos_id_map = {
            self.POS_TL: set_marker_id,
            self.POS_TR: self.FIXED_IDS[self.POS_TR],
            self.POS_BL: self.FIXED_IDS[self.POS_BL],
            self.POS_BR: self.FIXED_IDS[self.POS_BR],
            # Middle markers for estimation help
            self.POS_ML: self.FIXED_IDS[self.POS_ML],
            self.POS_MR: self.FIXED_IDS[self.POS_MR],
            # Fallback bottom markers
            self.POS_BL2: self.FIXED_IDS[self.POS_BL2],
            self.POS_BR2: self.FIXED_IDS[self.POS_BR2],
        }
        
        # -------------------------------------------------
        # 4. Identify which logical corners are visible
        # -------------------------------------------------
        # A logical corner is considered visible if either the primary or fallback marker is seen.
        logical_corners = set()
        
        # Check TL
        if pos_id_map[self.POS_TL] in detected_data:
            logical_corners.add(self.POS_TL)
            
        # Check TR
        if pos_id_map[self.POS_TR] in detected_data:
            logical_corners.add(self.POS_TR)
            
        # Check BL (Logically visible if BL or BL2 is visible)
        if pos_id_map[self.POS_BL] in detected_data or pos_id_map[self.POS_BL2] in detected_data:
            logical_corners.add(self.POS_BL)
            
        # Check BR (Logically visible if BR or BR2 is visible)
        if pos_id_map[self.POS_BR] in detected_data or pos_id_map[self.POS_BR2] in detected_data:
            logical_corners.add(self.POS_BR)

        # Map all currently detected markers by position name
        current_data_by_pos = {}
        for pos, mid in pos_id_map.items():
            if mid in detected_data:
                current_data_by_pos[pos] = detected_data[mid]

        # -------------------------------------------------
        # 5. Extract detection region corners from visible markers
        # -------------------------------------------------
        # Each corner of the detection region comes from a specific corner of a marker:
        # - Macropad TL = TL marker's top-right corner (index 1)
        # - Macropad TR = TR marker's top-left corner (index 0)
        # - Macropad BL = BL marker's top-left (index 0) OR BL2's bottom-right (index 2)
        # - Macropad BR = BR marker's top-right (index 1) OR BR2's bottom-left (index 3)

        region_corners = {}  # 'TL', 'TR', 'BL', 'BR' -> point

        # TL corner of detection region
        if pos_id_map[self.POS_TL] in detected_data:
            region_corners['TL'] = detected_data[pos_id_map[self.POS_TL]][1][1]

        # TR corner of detection region
        if pos_id_map[self.POS_TR] in detected_data:
            region_corners['TR'] = detected_data[pos_id_map[self.POS_TR]][1][0]

        # BL corner of detection region (from BL or BL2)
        bl_marker_id = pos_id_map[self.POS_BL]
        bl2_marker_id = pos_id_map[self.POS_BL2]
        if bl_marker_id in detected_data:
            region_corners['BL'] = detected_data[bl_marker_id][1][0]
        elif bl2_marker_id in detected_data:
            region_corners['BL'] = detected_data[bl2_marker_id][1][2]

        # BR corner of detection region (from BR or BR2)
        br_marker_id = pos_id_map[self.POS_BR]
        br2_marker_id = pos_id_map[self.POS_BR2]
        if br_marker_id in detected_data:
            region_corners['BR'] = detected_data[br_marker_id][1][1]
        elif br2_marker_id in detected_data:
            region_corners['BR'] = detected_data[br2_marker_id][1][3]

        # -------------------------------------------------
        # 6. Check how many corners we have and estimate if needed
        # -------------------------------------------------
        self._estimated_pos_names = set()

        # Get middle marker edge points for estimation (if visible)
        # Left edge: TL corner 1 → ML corner 1 (top-right) → BL corner 0
        # Right edge: TR corner 0 → MR corner 0 (top-left) → BR corner 1
        ml_corners = detected_data[pos_id_map[self.POS_ML]][1] if pos_id_map[self.POS_ML] in detected_data else None
        mr_corners = detected_data[pos_id_map[self.POS_MR]][1] if pos_id_map[self.POS_MR] in detected_data else None

        if len(region_corners) == 4:
            # All 4 corners visible - update cache
            self._marker_cache.clear()
            for pos, data in current_data_by_pos.items():
                self._marker_cache[pos] = data

        elif len(region_corners) == 3:
            # 3 corners visible - estimate the 4th
            missing = [k for k in ['TL', 'TR', 'BL', 'BR'] if k not in region_corners][0]
            estimated = None

            # Priority 1: Use middle marker for better accuracy (handles perspective)
            # Layout: TL → ML → BL2 → BL (vertically stacked on left)
            #         TR → MR → BR2 → BR (vertically stacked on right)
            # ML/MR bottom edge is roughly at the midpoint between top and bottom of detection area
            # So use the BOTTOM edge of middle markers (corner 2 for ML, corner 3 for MR)
            if missing == 'BL' and ml_corners is not None and 'TL' in region_corners:
                # Use ML's bottom-right corner (index 2) - on the left edge, at bottom of ML
                ml_bottom_edge = ml_corners[2]
                marker_size = np.linalg.norm(ml_corners[0] - ml_corners[1])
                estimated = 2 * ml_bottom_edge - region_corners['TL'] - np.array([0, marker_size]) 
            elif missing == 'BR' and mr_corners is not None and 'TR' in region_corners:
                # Use MR's bottom-left corner (index 3) - on the right edge, at bottom of MR
                mr_bottom_edge = mr_corners[3]
                marker_size = np.linalg.norm(mr_corners[0] - mr_corners[1])
                print(type(mr_bottom_edge))
                estimated = 2 * mr_bottom_edge - region_corners['TR'] - np.array([0, marker_size])
            elif missing == 'TL' and ml_corners is not None and 'BL' in region_corners:
                # Use ML's bottom-right corner (index 2)
                ml_bottom_edge = ml_corners[2]
                estimated = 2 * ml_bottom_edge - region_corners['BL']
            elif missing == 'TR' and mr_corners is not None and 'BR' in region_corners:
                # Use MR's bottom-left corner (index 3)
                mr_bottom_edge = mr_corners[3]
                estimated = 2 * mr_bottom_edge - region_corners['BR']

            # Priority 2: Fall back to parallelogram (less accurate with perspective)
            if estimated is None:
                if missing == 'TL':
                    estimated = region_corners['TR'] + region_corners['BL'] - region_corners['BR']
                elif missing == 'TR':
                    estimated = region_corners['TL'] + region_corners['BR'] - region_corners['BL']
                elif missing == 'BL':
                    estimated = region_corners['TL'] + region_corners['BR'] - region_corners['TR']
                elif missing == 'BR':
                    estimated = region_corners['TR'] + region_corners['BL'] - region_corners['TL']

            region_corners[missing] = estimated

            # Mark as estimated
            pos_name_map = {'TL': self.POS_TL, 'TR': self.POS_TR, 'BL': self.POS_BL, 'BR': self.POS_BR}
            self._estimated_pos_names = {pos_name_map[missing]}

        else:
            # Less than 3 corners, skip
            self._detection_valid = False
            return False

        # Build detection region array
        tl_corner = region_corners['TL']
        tr_corner = region_corners['TR']
        br_corner = region_corners['BR']
        bl_corner = region_corners['BL']

        detection_region = np.array([tl_corner, tr_corner, br_corner, bl_corner], dtype=np.float32)
        
        # -------------------------------------------------
        # 7. Compute variables for result
        # -------------------------------------------------
        grid_cells = self._compute_grid_cells(detection_region)
        
        marker_positions = {}
        for pos, (center, _) in current_data_by_pos.items():
            mid = pos_id_map[pos]
            marker_positions[mid] = center
            
        estimated_ids = {pos_id_map[p] for p in self._estimated_pos_names}

        self._last_detection = MacroPadDetection(
            set_marker_id=set_marker_id,
            detection_region=detection_region,
            grid_cells=grid_cells,
            marker_positions=marker_positions,
            estimated_markers=estimated_ids,
        )
        
        self._detection_valid = True
        return True
    
    def _compute_grid_cells(self, region: np.ndarray) -> List[np.ndarray]:
        """
        Compute 4x2 grid cell polygons from detection region.
        
        Args:
            region: 4 corners of detection region (TL, TR, BR, BL)
            
        Returns:
            List of 8 cell polygons (row-major order)
        """
        tl, tr, br, bl = region
        
        cells = []
        
        for row in range(self.GRID_ROWS):
            for col in range(self.GRID_COLS):
                # Compute cell corners using bilinear interpolation
                u0 = col / self.GRID_COLS
                u1 = (col + 1) / self.GRID_COLS
                v0 = row / self.GRID_ROWS
                v1 = (row + 1) / self.GRID_ROWS
                
                def interp(u, v):
                    top = tl + u * (tr - tl)
                    bottom = bl + u * (br - bl)
                    return top + v * (bottom - top)
                
                cell_tl = interp(u0, v0)
                cell_tr = interp(u1, v0)
                cell_br = interp(u1, v1)
                cell_bl = interp(u0, v1)
                
                cells.append(np.array([cell_tl, cell_tr, cell_br, cell_bl], dtype=np.float32))
        
        return cells
    
    def get_button_at_point(self, point: Tuple[float, float]) -> Optional[int]:
        """
        Get button index at the given point.
        
        Args:
            point: (x, y) in camera coordinates
            
        Returns:
            Button index (0-7) or None if not in any cell
        """
        if not self._detection_valid or self._last_detection is None:
            return None
        
        for idx, cell in enumerate(self._last_detection.grid_cells):
            result = cv2.pointPolygonTest(cell, point, False)
            if result >= 0:
                return idx
        
        return None
    
    def is_point_in_region(self, point: Tuple[float, float]) -> bool:
        """Check if point is within detection region."""
        if not self._detection_valid or self._last_detection is None:
            return False
        
        result = cv2.pointPolygonTest(
            self._last_detection.detection_region, point, False
        )
        return result >= 0
    
    @property
    def is_valid(self) -> bool:
        """Check if detection is valid."""
        return self._detection_valid
    
    @property
    def current_set_id(self) -> Optional[int]:
        """Get current detected set marker ID."""
        if self._last_detection:
            return self._last_detection.set_marker_id
        return None
    
    @property
    def detection(self) -> Optional[MacroPadDetection]:
        """Get last detection result."""
        return self._last_detection
    
    def draw_debug(
        self,
        frame: np.ndarray,
        finger_pos: Optional[Tuple[float, float]] = None,
        hovered_button: Optional[int] = None,
        activated_button: Optional[int] = None,
        set_name: str = ""
    ) -> np.ndarray:
        """
        Draw debug visualization.
        
        Args:
            frame: BGR frame to draw on
            finger_pos: Optional finger position
            hovered_button: Currently hovered button index
            activated_button: Just activated button index
            set_name: Name of current set
            
        Returns:
            Frame with debug overlay
        """
        output = frame.copy()
        
        if not self._detection_valid or self._last_detection is None:
            cv2.putText(output, "MacroPad: Not detected", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return output
        
        det = self._last_detection
        
        # Draw detection region
        region_pts = det.detection_region.astype(np.int32)
        cv2.polylines(output, [region_pts], True, (0, 255, 0), 2)
        
        # Draw grid cells
        for idx, cell in enumerate(det.grid_cells):
            cell_pts = cell.astype(np.int32)
            
            # Determine color based on state
            if idx == activated_button:
                color = (0, 255, 0)  # Green for activated
                thickness = 3
            elif idx == hovered_button:
                color = (0, 255, 255)  # Yellow for hovered
                thickness = 2
            else:
                color = (255, 128, 0)  # Orange for normal
                thickness = 1
            
            cv2.polylines(output, [cell_pts], True, color, thickness)
            
            # Draw button number
            center = np.mean(cell, axis=0).astype(int)
            cv2.putText(output, str(idx + 1), (center[0] - 5, center[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw marker positions
        for mid, pos in det.marker_positions.items():
            is_estimated = mid in det.estimated_markers
            color = (0, 165, 255) if is_estimated else (255, 255, 0)  # Orange vs Yellow
            cv2.circle(output, tuple(pos.astype(int)), 5, color, -1)
            label = f"*{mid}" if is_estimated else str(mid)
            cv2.putText(output, label, (int(pos[0]) + 8, int(pos[1])),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw finger position
        if finger_pos:
            fx, fy = int(finger_pos[0]), int(finger_pos[1])
            in_region = self.is_point_in_region(finger_pos)
            color = (0, 255, 0) if in_region else (0, 0, 255)
            cv2.circle(output, (fx, fy), 8, color, 2)
        
        # Status text
        status = f"MacroPad: Set {det.set_marker_id}"
        if set_name:
            status += f" ({set_name})"
        if det.estimated_markers:
            status += f" | Est: {list(det.estimated_markers)}"
        cv2.putText(output, status, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return output
    
    def reset(self):
        """Reset detection state and cache."""
        self._last_detection = None
        self._detection_valid = False
        self._marker_cache.clear()
        self._estimated_pos_names.clear()
        self._current_set_marker_id = None
