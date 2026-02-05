# Copyright (c) 2026 Huynh Huy. All rights reserved.

"""
HandFlow Feature Engineering
===========================

Includes velocity, acceleration, finger angles, and bounding box features.
"""

from __future__ import annotations
import numpy as np


class FeatureEngineer:
    """
    Compute optimized features for hand gesture recognition.

    Features (96 total):
    - Relative Positions (63): Wrist-normalized (x, y, z) for 21 landmarks
    - Inter-finger Distances (5): Distances between thumb and fingertips
    - Raw Positions (9): Absolute coords for thumb tip, index MCP/tip
    - Velocity Features (9): Frame-to-frame motion of index MCP/tip and thumb tip (FPS-normalized)
    - Finger Angles (5): Bending angle at PIP joint per finger
    - Pinch Dynamics (3): Aperture delta/acceleration and thumb-index z-diff
    - Thumb Posture (2): Abduction angle and thumb-wrist distance

    FPS Normalization:
    - Velocity features are normalized by delta_time to be FPS-invariant
    - This ensures consistent model behavior regardless of frame rate differences
    - Reference FPS is read from config (data.target_fps)
    """

    # Default reference FPS (overridden by config if available)
    DEFAULT_REFERENCE_FPS = 20.0

    def __init__(self, reference_fps: float = None) -> None:
        """
        Initialize feature engineer.

        Args:
            reference_fps: Reference FPS for velocity normalization.
                          If None, tries to read from config, falls back to default.
        """
        if reference_fps is not None:
            self._reference_fps = reference_fps
        else:
            # Try to load from config
            try:
                from handflow.utils import load_config
                config = load_config("config/config.yaml")
                self._reference_fps = getattr(config.data, 'target_fps', self.DEFAULT_REFERENCE_FPS)
            except Exception:
                self._reference_fps = self.DEFAULT_REFERENCE_FPS

        self._reference_dt = 1.0 / self._reference_fps

    def transform(self, sequence: np.ndarray, delta_time: float = None) -> np.ndarray:
        """
        Transform a sequence of keypoints using optimized feature set.

        Args:
            sequence: Raw keypoints of shape (num_frames, 84).
            delta_time: Time between frames in seconds. If provided, velocities
                       are normalized to be FPS-invariant. If None, uses reference FPS.

        Returns:
            Optimized features of shape (num_frames, 93).
        """
        # Raw Positions (T, 63)
        # Reshape to (T, 21, 4) and take x,y,z
        landmarks_xyz = sequence.reshape(-1, 21, 4)[:, :, :3]

        # -------------------------------------------------
        # 1. Raw Features Extraction
        # -------------------------------------------------
        # Raw Thumb Tip (Landmark 4) (T, 3)
        raw_thumb_tip = landmarks_xyz[:, 4, :]

        # Raw Index MCP (Landmark 5) (T, 3)
        raw_index_mcp = landmarks_xyz[:, 5, :]

        # Raw Index Tip (Landmark 8) (T, 3)
        raw_index_tip = landmarks_xyz[:, 8, :]

        # -------------------------------------------------
        # 2. Velocity Features (FPS-normalized)
        # -------------------------------------------------
        # Velocity of Index MCP (Landmark 5)
        vel_index_mcp = np.diff(raw_index_mcp, axis=0, prepend=raw_index_mcp[0:1])

        # Velocity of Index Tip (Landmark 8)
        vel_index_tip = np.diff(raw_index_tip, axis=0, prepend=raw_index_tip[0:1])

        # Velocity of Thumb Tip (Landmark 4) - KEY for touch vs touch_hover
        # touch: thumb moves toward index (high velocity)
        # touch_hover: thumb is stationary (near-zero velocity)
        vel_thumb_tip = np.diff(raw_thumb_tip, axis=0, prepend=raw_thumb_tip[0:1])

        # Normalize velocities by delta time to make them FPS-invariant
        # This ensures consistent velocity magnitudes regardless of frame rate
        if delta_time is not None and delta_time > 0:
            # Scale velocities: v_normalized = v_raw * (reference_dt / actual_dt)
            # If running slower (larger dt), velocities per frame are larger,
            # so we scale down to match reference FPS behavior
            time_scale = self._reference_dt / delta_time
            vel_index_mcp = vel_index_mcp * time_scale
            vel_index_tip = vel_index_tip * time_scale
            vel_thumb_tip = vel_thumb_tip * time_scale

        # -------------------------------------------------
        # 3. Normalized Features
        # -------------------------------------------------
        # Normalize relative to Wrist (Landmark 0) for translation invariance
        wrist = landmarks_xyz[:, 0:1, :]
        relative_xyz = landmarks_xyz - wrist

        positions = relative_xyz.reshape(sequence.shape[0], -1)  # (T, 63)

        # -------------------------------------------------
        # 4. Inter-finger Distances (T, 5)
        # -------------------------------------------------
        distances = self._compute_inter_finger_distances(sequence)

        # -------------------------------------------------
        # 5. Finger Angles (T, 5)
        # -------------------------------------------------
        angles = self._compute_finger_angles(sequence)

        # -------------------------------------------------
        # 6. Pinch Dynamics Features - KEY for touch vs touch_hover
        # -------------------------------------------------
        # Thumb-Index distance delta (rate of change)
        # touch: distance rapidly DECREASES then INCREASES (tap pattern)
        # touch_hover: distance is STABLE (near-zero delta)
        thumb_index_dist = distances[:, 0:1]  # (T, 1)
        pinch_aperture_delta = np.diff(thumb_index_dist, axis=0, prepend=thumb_index_dist[0:1])  # (T, 1)

        # Pinch aperture acceleration (second derivative)
        # touch: has sharp "snap" - high acceleration at moment of contact
        # touch_hover: maintains constant aperture - near-zero acceleration
        pinch_aperture_accel = np.diff(pinch_aperture_delta, axis=0, prepend=pinch_aperture_delta[0:1])  # (T, 1)

        # Thumb-Index Z-difference (depth alignment)
        # touch: tips are at SAME depth (z-diff ≈ 0)
        # touch_hover: thumb is typically BEHIND index (larger z-diff)
        thumb_index_z_diff = (raw_thumb_tip[:, 2:3] - raw_index_tip[:, 2:3])  # (T, 1)

        # -------------------------------------------------
        # 7. Thumb Posture Features - KEY for touch vs touch_hover
        # -------------------------------------------------
        wrist_flat = landmarks_xyz[:, 0, :]  # (T, 3)
        index_mcp = landmarks_xyz[:, 5, :]   # (T, 3)

        # Thumb Abduction Angle (thumb openness relative to palm)
        # touch: thumb ADDUCTED (closer to palm/index)
        # touch_hover: thumb ABDUCTED (open, away from palm)
        v_thumb = raw_thumb_tip - wrist_flat  # (T, 3)
        v_palm = index_mcp - wrist_flat       # (T, 3)
        dot_prod = np.sum(v_thumb * v_palm, axis=1, keepdims=True)
        norm_thumb = np.linalg.norm(v_thumb, axis=1, keepdims=True) + 1e-8
        norm_palm = np.linalg.norm(v_palm, axis=1, keepdims=True) + 1e-8
        thumb_abduction = np.arccos(np.clip(dot_prod / (norm_thumb * norm_palm), -1, 1))  # (T, 1)

        # Thumb-Wrist Distance (thumb extension)
        # touch_hover: thumb is curled back toward wrist
        # touch: thumb is extended toward index
        thumb_wrist_dist = np.linalg.norm(raw_thumb_tip - wrist_flat, axis=1, keepdims=True)  # (T, 1)

        # Pre-allocate output array and fill in-place (faster than concatenate)
        # [Positions(63) | Distances(5) | RawThumb(3) | RawIndexMCP(3) | RawIndexTip(3) | 
        #  VelIndexMCP(3) | VelIndexTip(3) | Angles(5) | VelThumbTip(3) | PinchDelta(1) | 
        #  PinchAccel(1) | ThumbIndexZDiff(1) | ThumbAbduction(1) | ThumbWristDist(1)]
        # Total: 63 + 5 + 3 + 3 + 3 + 3 + 3 + 5 + 3 + 1 + 1 + 1 + 1 + 1 = 96
        T = sequence.shape[0]
        output = np.empty((T, 96), dtype=np.float32)

        output[:, 0:63] = positions
        output[:, 63:68] = distances
        output[:, 68:71] = raw_thumb_tip
        output[:, 71:74] = raw_index_mcp
        output[:, 74:77] = raw_index_tip
        output[:, 77:80] = vel_index_mcp
        output[:, 80:83] = vel_index_tip
        output[:, 83:88] = angles
        output[:, 88:91] = vel_thumb_tip
        output[:, 91:92] = pinch_aperture_delta
        output[:, 92:93] = pinch_aperture_accel
        output[:, 93:94] = thumb_index_z_diff
        output[:, 94:95] = thumb_abduction
        output[:, 95:96] = thumb_wrist_dist

        return output

    def _compute_inter_finger_distances(self, sequence: np.ndarray) -> np.ndarray:
        """
        Compute distances between adjacent fingertips over time.
    
        Returns:
            distances: (T, 4)
        """
        num_frames = sequence.shape[0]
        distances = np.zeros((num_frames, 5))
    
        # Fingertip landmark indices: Thumb to Index, Mid, Ring, Pinky
        pairs = [(4, 8), (4, 12), (4, 16), (4, 20), (4, 6)]
    
        for i, (p1, p2) in enumerate(pairs):
            p1_coords = sequence[:, p1*4 : p1*4 + 3]
            p2_coords = sequence[:, p2*4 : p2*4 + 3]
            distances[:, i] = np.linalg.norm(p1_coords - p2_coords, axis=1)
    
        return distances

    def _compute_finger_angles(self, sequence: np.ndarray) -> np.ndarray:
        """
        Compute bending angles for all 5 fingers.
        Angle is computed at the middle joint (PIP for fingers, IP for thumb).
        Returns: (T, 5) where values are normalized radians (or degrees scaled).
        """
        # Indices for joints [Proxim, Mid, Distal]
        # Thumb: 2, 3, 4
        # Index: 5, 6, 7
        # Middle: 9, 10, 11
        # Ring: 13, 14, 15
        # Pinky: 17, 18, 19
        
        joints = [
            (2, 3, 4),   # Thumb
            (5, 6, 7),   # Index
            (9, 10, 11), # Middle
            (13, 14, 15),# Ring
            (17, 18, 19) # Pinky
        ]
        
        T = sequence.shape[0]
        angles = np.zeros((T, 5))
        
        for i, (a_idx, b_idx, c_idx) in enumerate(joints):
            # Extract points (T, 3)
            # Reshape raw features to (T, 21, 4) then take :3 => (T, 21, 3) if input is (T, 84)
            # But sequence input is (T, 84) flattened.
            # So:
            a = sequence[:, a_idx*4 : a_idx*4+3]
            b = sequence[:, b_idx*4 : b_idx*4+3]
            c = sequence[:, c_idx*4 : c_idx*4+3]
            
            # Vectors BA and BC
            ba = a - b
            bc = c - b
            
            # Normalize vectors
            # Add eps to avoid div by zero
            norm_ba = np.linalg.norm(ba, axis=1, keepdims=True) + 1e-8
            norm_bc = np.linalg.norm(bc, axis=1, keepdims=True) + 1e-8
            
            dot_prod = np.sum(ba * bc, axis=1, keepdims=True) / (norm_ba * norm_bc)
            
            # Clip for arccos stability
            dot_prod = np.clip(dot_prod, -1.0, 1.0)
            
            # Angle in radians
            angle_rad = np.arccos(dot_prod)
            
            # Store in output (flatten from (T,1) to (T,))
            angles[:, i] = angle_rad.flatten()
            
        return angles

    def get_output_dim(self) -> int:
        """
        Get the output dimension after feature engineering.

        Returns:
            Total feature count: 96
        """
        return 96


def add_velocity_features(
    keypoints: np.ndarray,
    delta_time: float = None,
    reference_fps: float = 20.0
) -> np.ndarray:
    """
    Standalone function to add velocity features with optional FPS normalization.

    Args:
        keypoints: Shape (num_frames, 84).
        delta_time: Time between frames in seconds. If provided, normalizes velocities.
        reference_fps: Reference FPS for normalization (default 20).

    Returns:
        Array with velocity appended, shape (num_frames, 168).
    """
    velocities = np.diff(keypoints, axis=0, prepend=keypoints[0:1])

    # Normalize if delta_time provided
    if delta_time is not None and delta_time > 0:
        reference_dt = 1.0 / reference_fps
        time_scale = reference_dt / delta_time
        velocities = velocities * time_scale

    return np.concatenate([keypoints, velocities], axis=-1)


def add_acceleration_features(
    keypoints: np.ndarray,
    delta_time: float = None,
    reference_fps: float = 20.0
) -> np.ndarray:
    """
    Standalone function to add acceleration features with optional FPS normalization.

    Args:
        keypoints: Shape (num_frames, 84).
        delta_time: Time between frames in seconds. If provided, normalizes.
        reference_fps: Reference FPS for normalization (default 20).

    Returns:
        Array with acceleration appended, shape (num_frames, 168).
    """
    velocity = np.diff(keypoints, axis=0, prepend=keypoints[0:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])

    # Normalize if delta_time provided
    if delta_time is not None and delta_time > 0:
        reference_dt = 1.0 / reference_fps
        time_scale = reference_dt / delta_time
        # Acceleration scales with time^2
        velocity = velocity * time_scale
        acceleration = acceleration * (time_scale ** 2)

    return np.concatenate([keypoints, acceleration], axis=-1)
