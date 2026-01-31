"""
Tests for Model Architectures
=============================
"""

import numpy as np
import pytest


class TestArchitectures:
    """Tests for model architecture builders."""

    def test_build_lstm_model(self, mock_config):
        """Test LSTM model building."""
        from handflow.models import build_lstm_model

        model = build_lstm_model(mock_config)

        assert model is not None
        assert "lstm" in model.name.lower()

        # Check input/output shapes
        input_shape = model.input_shape
        output_shape = model.output_shape

        assert input_shape[1] == mock_config.model.sequence_length
        assert output_shape[1] == mock_config.model.num_classes

    def test_build_gru_model(self, mock_config):
        """Test GRU model building."""
        from handflow.models import build_gru_model

        model = build_gru_model(mock_config)

        assert model is not None
        assert "gru" in model.name.lower()

    def test_build_cnn1d_model(self, mock_config):
        """Test 1D CNN model building."""
        from handflow.models import build_cnn1d_model

        model = build_cnn1d_model(mock_config)

        assert model is not None
        assert "cnn" in model.name.lower()

    def test_build_transformer_model(self, mock_config):
        """Test Transformer model building."""
        from handflow.models import build_transformer_model

        model = build_transformer_model(mock_config)

        assert model is not None
        assert "transformer" in model.name.lower()

    def test_build_model_factory(self, mock_config):
        """Test build_model factory function."""
        from handflow.models import build_model

        for arch in ["lstm", "gru", "cnn1d", "transformer"]:
            mock_config.model.architecture = arch
            model = build_model(mock_config)
            assert model is not None

    def test_build_model_invalid_architecture(self, mock_config):
        """Test build_model with invalid architecture."""
        from handflow.models import build_model

        mock_config.model.architecture = "invalid"

        with pytest.raises(ValueError):
            build_model(mock_config)

    def test_model_forward_pass(self, mock_config, sample_batch):
        """Test model forward pass."""
        from handflow.models import build_model

        x, y = sample_batch

        for arch in ["lstm", "gru", "cnn1d", "transformer"]:
            mock_config.model.architecture = arch
            model = build_model(mock_config)

            output = model.predict(x, verbose=0)

            assert output.shape == (x.shape[0], mock_config.model.num_classes)
            # Softmax output should sum to ~1
            np.testing.assert_array_almost_equal(
                output.sum(axis=1), np.ones(x.shape[0]), decimal=5
            )


class TestCountParameters:
    """Tests for parameter counting."""

    def test_count_parameters(self, mock_config):
        """Test parameter counting utility."""
        from handflow.models import build_model, count_parameters

        mock_config.model.architecture = "gru"
        model = build_model(mock_config)

        params = count_parameters(model)

        assert "trainable" in params
        assert "non_trainable" in params
        assert "total" in params
        assert params["trainable"] > 0
        assert params["total"] == params["trainable"] + params["non_trainable"]
