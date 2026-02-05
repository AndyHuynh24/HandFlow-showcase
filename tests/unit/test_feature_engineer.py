# Copyright (c) 2026 Huynh Huy. All rights reserved.

"""
Tests for Feature Engineering
=============================
"""

import numpy as np
import pytest


class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""

    def test_velocity_features(self, sample_sequence, mock_config):
        """Test velocity feature computation."""
        from handflow.features import FeatureEngineer

        # Enable only velocity
        mock_config.features.velocity = True
        mock_config.features.acceleration = False
        mock_config.features.finger_angles = False
        mock_config.features.hand_bbox_size = False

        engineer = FeatureEngineer(mock_config)
        result = engineer.transform(sample_sequence)

        # Should double the feature dimension (84 + 84 velocity)
        assert result.shape == (16, 168)

    def test_all_features(self, sample_sequence, mock_config):
        """Test with all features enabled."""
        from handflow.features import FeatureEngineer

        mock_config.features.velocity = True
        mock_config.features.acceleration = True
        mock_config.features.finger_angles = True
        mock_config.features.hand_bbox_size = True

        engineer = FeatureEngineer(mock_config)
        result = engineer.transform(sample_sequence)

        # 84 + 84 (velocity) + 84 (acceleration) + 15 (angles) + 4 (bbox)
        expected_dim = 84 + 84 + 84 + 15 + 4
        assert result.shape == (16, expected_dim)

    def test_output_dim_calculation(self, mock_config):
        """Test get_output_dim method."""
        from handflow.features import FeatureEngineer

        mock_config.features.velocity = True
        mock_config.features.acceleration = True
        mock_config.features.finger_angles = True
        mock_config.features.hand_bbox_size = True

        engineer = FeatureEngineer(mock_config)
        dim = engineer.get_output_dim()

        assert dim == 84 + 84 + 84 + 15 + 4

    def test_no_features(self, sample_sequence, mock_config):
        """Test with no additional features."""
        from handflow.features import FeatureEngineer

        mock_config.features.velocity = False
        mock_config.features.acceleration = False
        mock_config.features.finger_angles = False
        mock_config.features.hand_bbox_size = False

        engineer = FeatureEngineer(mock_config)
        result = engineer.transform(sample_sequence)

        # Should be unchanged
        assert result.shape == sample_sequence.shape
        np.testing.assert_array_equal(result, sample_sequence)

    def test_finger_angles_range(self, sample_sequence, mock_config):
        """Test that finger angles are in [0, 1] range."""
        from handflow.features import FeatureEngineer

        mock_config.features.velocity = False
        mock_config.features.acceleration = False
        mock_config.features.finger_angles = True
        mock_config.features.hand_bbox_size = False

        engineer = FeatureEngineer(mock_config)
        result = engineer.transform(sample_sequence)

        # Angles are the last 15 features
        angles = result[:, 84:]
        assert angles.min() >= 0
        assert angles.max() <= 1


class TestStandaloneFunctions:
    """Tests for standalone feature functions."""

    def test_add_velocity_features(self, sample_sequence):
        """Test standalone velocity function."""
        from handflow.features import add_velocity_features

        result = add_velocity_features(sample_sequence)
        assert result.shape == (16, 168)

    def test_add_acceleration_features(self, sample_sequence):
        """Test standalone acceleration function."""
        from handflow.features import add_acceleration_features

        result = add_acceleration_features(sample_sequence)
        assert result.shape == (16, 168)
