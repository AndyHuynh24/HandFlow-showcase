"""
Pytest Configuration and Fixtures
==================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np
import pytest


@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_keypoints() -> np.ndarray:
    """Generate sample keypoint data (single frame)."""
    # 21 landmarks * 4 values (x, y, z, visibility)
    return np.random.randn(84).astype(np.float32)


@pytest.fixture
def sample_sequence() -> np.ndarray:
    """Generate sample keypoint sequence (16 frames)."""
    return np.random.randn(16, 84).astype(np.float32)


@pytest.fixture
def sample_batch(mock_config) -> tuple[np.ndarray, np.ndarray]:
    """Generate a batch of sequences with labels."""
    batch_size = 32
    seq_len = mock_config.model.sequence_length
    input_dim = mock_config.get_input_dim()  # Use config's calculated dim
    num_classes = mock_config.model.num_classes

    x = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)
    y = np.zeros((batch_size, num_classes), dtype=np.float32)
    y[np.arange(batch_size), np.random.randint(0, num_classes, batch_size)] = 1

    return x, y


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    from handflow.utils.config import Config

    config = Config()
    config.model.sequence_length = 16
    config.model.input_dim = 84
    config.model.num_classes = 8
    config.model.hidden_units = 64  # Smaller for testing
    config.model.dropout = 0.2

    config.training.epochs = 2  # Quick training
    config.training.batch_size = 8

    config.training.epochs = 2  # Quick training
    config.training.batch_size = 8

    return config

    return config


@pytest.fixture
def sample_landmarks() -> np.ndarray:
    """Generate sample hand landmarks (21 points, 4 values each)."""
    landmarks = np.random.randn(21, 4).astype(np.float32)
    # Normalize x, y to [0, 1]
    landmarks[:, 0] = np.clip(landmarks[:, 0], 0, 1)
    landmarks[:, 1] = np.clip(landmarks[:, 1], 0, 1)
    landmarks[:, 2] = landmarks[:, 2] * 0.01  # Small z values
    landmarks[:, 3] = 1.0  # Visibility
    return landmarks.flatten()


@pytest.fixture
def temp_model_path(tmp_path: Path, mock_config) -> Generator[Path, None, None]:
    """Create a temporary model for testing."""
    from handflow.models import build_model

    model = build_model(mock_config)
    model_path = tmp_path / "test_model.h5"
    model.save(str(model_path))

    yield model_path

    # Cleanup handled by tmp_path fixture


@pytest.fixture
def gesture_actions() -> list[str]:
    """List of gesture action names."""
    return [
        "none",
        "nonezoom",
        "swiperight",
        "zoom",
        "pointyclick",
        "middleclick",
        "ringclick",
        "pinkyclick",
    ]
