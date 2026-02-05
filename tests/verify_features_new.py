# Copyright (c) 2026 Huynh Huy. All rights reserved.

import unittest
import numpy as np
from handflow.features.feature_engineer import FeatureEngineer
from handflow.utils.config import Config

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.engineer = FeatureEngineer(self.config)
        
        # Create a dummy sequence: 16 frames, 84 features (21 landmarks * 4 coords)
        self.num_frames = 16
        self.input_sequence = np.random.rand(self.num_frames, 84)

    def test_output_shape(self):
        """Test that the output shape is exactly (num_frames, 88)."""
        output = self.engineer.transform(self.input_sequence)
        self.assertEqual(output.shape, (self.num_frames, 88), 
                         f"Expected shape (16, 88), but got {output.shape}")

    def test_get_output_dim(self):
        """Test that get_output_dim returns 88."""
        self.assertEqual(self.engineer.get_output_dim(), 88)
    
    def test_pinch_distance_logic(self):
        """Verify pinch distance specific indices."""
        # This is a basic smoke test to ensure the method runs without error
        output = self.engineer._compute_inter_finger_distances(self.input_sequence)
        self.assertEqual(output.shape, (self.num_frames, 4))
        
    def test_get_input_dim_config(self):
        """Test that config.get_input_dim() returns 88."""
        self.assertEqual(self.config.get_input_dim(), 88)


if __name__ == '__main__':
    unittest.main()
