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
    
    Features (67 total):
    - Raw Positions (63): (x, y, z) for 21 landmarks
    - Pinch Distances (4): Distances between adjacent fingertips
    """

    def __init__(self) -> None:
        """
        Initialize feature engineer.
        """

    def transform(self, sequence: np.ndarray) -> np.ndarray:
        """
        Transform a sequence of keypoints using optimized feature set.

        Args:
            sequence: Raw keypoints of shape (num_frames, 84).

        Returns:
            Optimized features of shape (num_frames, 88).
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
        # 2. Velocity Features
        # -------------------------------------------------
        # Velocity of Index MCP (Landmark 5)
        vel_index_mcp = np.diff(raw_index_mcp, axis=0, prepend=raw_index_mcp[0:1])

        # Velocity of Index Tip (Landmark 8)
        vel_index_tip = np.diff(raw_index_tip, axis=0, prepend=raw_index_tip[0:1])

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

        # Concatenate all features:
        # [Positions(63) | Distances(4) | RawThumb(3) | RawIndexMCP(3) | RawIndexTip(3) | VelIndexMCP(3) | VelIndexTip(3) | Angles(5)]
        # Total: 63 + 4 + 3 + 3 + 3 + 3 + 3 + 5 = 88
        return np.concatenate([
            positions,
            distances,
            raw_thumb_tip,
            raw_index_mcp,
            raw_index_tip,
            vel_index_mcp,
            vel_index_tip,
            angles
        ], axis=-1)

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
            Total feature count: 88
        """
        return 88


def add_velocity_features(keypoints: np.ndarray) -> np.ndarray:
    """
    Standalone function to add velocity features.

    Args:
        keypoints: Shape (num_frames, 84).

    Returns:
        Array with velocity appended, shape (num_frames, 168).
    """
    velocities = np.diff(keypoints, axis=0, prepend=keypoints[0:1])
    return np.concatenate([keypoints, velocities], axis=-1)


def add_acceleration_features(keypoints: np.ndarray) -> np.ndarray:
    """
    Standalone function to add acceleration features.

    Args:
        keypoints: Shape (num_frames, 84).

    Returns:
        Array with acceleration appended, shape (num_frames, 168).
    """
    velocity = np.diff(keypoints, axis=0, prepend=keypoints[0:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
    return np.concatenate([keypoints, acceleration], axis=-1)
