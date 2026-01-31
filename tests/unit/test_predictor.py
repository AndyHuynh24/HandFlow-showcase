"""
Tests for Gesture Predictor
===========================
"""

import numpy as np
import pytest


class TestGesturePredictor:
    """Tests for GesturePredictor class."""

    def test_add_frame(self, sample_keypoints, mock_config, temp_model_path, gesture_actions):
        """Test adding frames to buffer."""
        from handflow.models import GesturePredictor

        predictor = GesturePredictor(
            model_path=temp_model_path,
            config=mock_config,
            actions=gesture_actions,
        )

        # Buffer should be empty initially
        assert len(predictor.sequence_buffer) == 0
        assert not predictor.is_ready()

        # Add a frame
        predictor.add_frame(sample_keypoints)
        assert len(predictor.sequence_buffer) == 1

    def test_is_ready(self, sample_keypoints, mock_config, temp_model_path, gesture_actions):
        """Test buffer ready check."""
        from handflow.models import GesturePredictor

        predictor = GesturePredictor(
            model_path=temp_model_path,
            config=mock_config,
            actions=gesture_actions,
        )

        # Add frames until ready
        for _ in range(15):
            predictor.add_frame(sample_keypoints)
            assert not predictor.is_ready()

        predictor.add_frame(sample_keypoints)
        assert predictor.is_ready()

    def test_predict(self, sample_keypoints, mock_config, temp_model_path, gesture_actions):
        """Test prediction."""
        from handflow.models import GesturePredictor

        predictor = GesturePredictor(
            model_path=temp_model_path,
            config=mock_config,
            actions=gesture_actions,
        )

        # Fill buffer
        for _ in range(16):
            predictor.add_frame(sample_keypoints)

        # Get prediction
        gesture, confidence, probs = predictor.predict()

        # Gesture might be None (cooldown, stability filter)
        # But probs should always be valid
        assert probs.shape == (len(gesture_actions),)
        assert 0 <= confidence <= 1

    def test_reset(self, sample_keypoints, mock_config, temp_model_path, gesture_actions):
        """Test reset method."""
        from handflow.models import GesturePredictor

        predictor = GesturePredictor(
            model_path=temp_model_path,
            config=mock_config,
            actions=gesture_actions,
        )

        # Add some frames
        for _ in range(10):
            predictor.add_frame(sample_keypoints)

        predictor.reset()

        assert len(predictor.sequence_buffer) == 0
        assert len(predictor.prediction_history) == 0


class TestDualHandPredictor:
    """Tests for DualHandPredictor class."""

    def test_initialization(self, mock_config, tmp_path):
        """Test dual hand predictor initialization."""
        # This test would need actual model files
        # Skip if models don't exist
        pytest.skip("Requires actual model files")
