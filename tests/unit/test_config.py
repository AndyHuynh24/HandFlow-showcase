# Copyright (c) 2026 Huynh Huy. All rights reserved.

"""
Tests for Configuration
=======================
"""

from pathlib import Path

import pytest


class TestConfig:
    """Tests for configuration management."""

    def test_default_config(self):
        """Test default configuration values."""
        from handflow.utils.config import Config

        config = Config()

        assert config.model.architecture == "gru"
        assert config.model.sequence_length == 16
        assert config.training.epochs == 100
        assert config.features.velocity is True

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        from handflow.utils.config import Config

        data = {
            "model": {
                "architecture": "lstm",
                "hidden_units": 256,
            },
            "training": {
                "epochs": 50,
                "batch_size": 64,
            },
            "features": {
                "velocity": False,
                "acceleration": True,
            },
        }

        config = Config.from_dict(data)

        assert config.model.architecture == "lstm"
        assert config.model.hidden_units == 256
        assert config.training.epochs == 50
        assert config.training.batch_size == 64
        assert config.features.velocity is False
        assert config.features.acceleration is True

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        from handflow.utils.config import Config

        config = Config()
        data = config.to_dict()

        assert isinstance(data, dict)
        assert "model" in data
        assert "training" in data
        assert "features" in data

    def test_get_input_dim(self):
        """Test input dimension calculation."""
        from handflow.utils.config import Config

        config = Config()

        # All features enabled
        config.features.velocity = True
        config.features.acceleration = True
        config.features.finger_angles = True
        config.features.hand_bbox_size = True

        # 84 + 84 + 84 + 15 + 4 = 271
        assert config.get_input_dim() == 271

        # Only base features
        config.features.velocity = False
        config.features.acceleration = False
        config.features.finger_angles = False
        config.features.hand_bbox_size = False

        assert config.get_input_dim() == 84

    def test_load_config_default(self, project_root):
        """Test loading default config."""
        from handflow.utils.config import load_config

        # Should not raise even if file doesn't exist
        config = load_config(project_root / "config" / "config.yaml")
        assert config is not None

    def test_gestures_default(self):
        """Test default gesture lists."""
        from handflow.utils.config import Config

        config = Config()

        assert len(config.right_hand_gestures) == 8
        assert len(config.left_hand_gestures) == 8
        assert "none" in config.right_hand_gestures
        assert "swipeleft" in config.left_hand_gestures
