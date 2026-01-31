"""
HandFlow Model Architectures
============================

Neural network architectures for gesture classification.
Supports TCN, LSTM, GRU, 1D-CNN, and Transformer models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

if TYPE_CHECKING:
    from handflow.utils.config import Config

@tf.keras.utils.register_keras_serializable(package="HandFlow")
class SoftMotionWeighting(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        # x: (B, T, C)
        velocity = tf.norm(x[:, 1:] - x[:, :-1], axis=-1, keepdims=True)
        velocity = tf.pad(velocity, [[0, 0], [1, 0], [0, 0]])
        weights = tf.sigmoid(velocity * 5.0)
        return x * weights

    def get_config(self):
        return super().get_config()


def build_model(config: Config) -> keras.Model:
    """
    Build a model based on configuration.

    Args:
        config: Configuration with model settings.

    Returns:
        Compiled Keras model.
    """
    architecture = config.model.architecture.lower()

    builders = {
        "lstm": build_lstm_model,
        "gru": build_gru_model,
        "cnn1d": build_cnn1d_model,
        "tcn": build_tcn_model,
        "transformer": build_transformer_model,
    }

    if architecture not in builders:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Available: {list(builders.keys())}"
        )

    model = builders[architecture](config)

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.training.learning_rate),
        loss=config.training.loss,
        metrics=["accuracy"],
    )

    return model


def build_lstm_model(config: Config) -> keras.Model:
    """
    Build LSTM-based gesture classifier.

    Args:
        config: Configuration object.

    Returns:
        Keras Sequential model.
    """
    input_dim = config.model.input_dim()
    seq_len = config.data.sequence_length
    hidden_units = config.model.hidden_units
    dropout = config.model.dropout
    num_classes = config.model.num_classes

    model = keras.Sequential(
        [
            layers.Input(shape=(seq_len, input_dim)),
            
            layers.LSTM(hidden_units, return_sequences=True),
            layers.Dropout(dropout),
            
            layers.LSTM(hidden_units // 2),
            layers.Dropout(dropout),
            
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout / 2),
            layers.Dense(num_classes, activation="softmax"),
        ],
        # name="lstm_gesture_classifier",
    )

    
    

    return model


def build_gru_model(config: Config) -> keras.Model:
    """
    Build GRU-based gesture classifier.

    Args:
        config: Configuration object.

    Returns:
        Keras Sequential model.
    """
    input_dim = config.model.input_dim()
    seq_len = config.data.sequence_length
    hidden_units = config.model.hidden_units
    dropout = config.model.dropout
    num_classes = config.model.num_classes

    model = keras.Sequential(
        [
            layers.Input(shape=(seq_len, input_dim)),
         
            layers.GRU(hidden_units, return_sequences=True),
            layers.Dropout(dropout),
   
            layers.GRU(hidden_units // 2),
            layers.Dropout(dropout),
        
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout / 2),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="gru_gesture_classifier",
    )

    return model


def build_cnn1d_model(config: Config) -> keras.Model:
    """
    Build 1D CNN gesture classifier.

    Args:
        config: Configuration object.

    Returns:
        Keras Sequential model.
    """
    input_dim = config.model.input_dim
    seq_len = config.data.sequence_length
    dropout = config.model.dropout
    num_classes = config.model.num_classes

    model = keras.Sequential(
        [
            layers.Input(shape=(seq_len, input_dim)),
  
            layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
       
            layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
           
            layers.Conv1D(128, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
        
            layers.GlobalAveragePooling1D(),
            
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="cnn1d_gesture_classifier",
    )

    return model


def build_transformer_model(config: Config) -> keras.Model:
    """
    Build Transformer-based gesture classifier.

    Args:
        config: Configuration object.

    Returns:
        Keras Functional model.
    """
    input_dim = config.model.input_dim
    seq_len = config.data.sequence_length
    dropout = config.model.dropout
    num_classes = config.model.num_classes
    num_heads = 4
    head_dim = 64
    ff_dim = 128
    num_blocks = getattr(config.model, 'num_transformer_blocks', 2)

    inputs = layers.Input(shape=(seq_len, input_dim))

    # Project input to head_dim for attention
    x = layers.Dense(head_dim)(inputs)

    # Positional encoding (learned)
    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_embedding = layers.Embedding(input_dim=seq_len, output_dim=head_dim)(positions)
    x = x + pos_embedding

    # Stack Transformer blocks
    for _ in range(num_blocks):
        # Multi-head self-attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=head_dim, dropout=dropout
        )(x, x)
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # Feed-forward network
        ff_output = layers.Dense(ff_dim, activation="relu")(x)
        ff_output = layers.Dropout(dropout)(ff_output)
        ff_output = layers.Dense(head_dim)(ff_output)
        x = layers.Add()([x, ff_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Global average pooling (attend to entire sequence)
    x = layers.GlobalAveragePooling1D()(x)

    # Classification head
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="transformer_gesture_classifier")

    return model


def build_tcn_model(config: Config) -> keras.Model:
    """
    Build TCN-based gesture classifier with residual connections.

    Architecture:
    - Input projection to match filter dimension
    - Stacked dilated causal convolution blocks with residuals
    - Each block: Conv1D -> BatchNorm -> ReLU -> Conv1D -> Add residual
    - Global pooling + classification head

    Residual connections help by:
    1. Enabling gradient flow through skip connections (no vanishing gradients)
    2. Allowing layers to learn "refinements" instead of full transformations
    3. Making deeper networks trainable
    4. Improving convergence speed
    """
    input_dim = config.model.input_dim
    seq_len = config.data.sequence_length
    dropout = config.model.dropout
    num_classes = config.model.num_classes
    filters = 128

    inputs = layers.Input(shape=(seq_len, input_dim))

    # Project input to filter dimension for residual connections
    x = layers.Conv1D(filters, kernel_size=1, padding="same")(inputs)

    # TCN blocks with residual connections
    for dilation in [1, 2, 4]:
        x = _tcn_residual_block(x, filters, kernel_size=3, dilation=dilation, dropout=dropout)

    # Temporal aggregation: combine avg and max pooling for richer representation
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)
    x = layers.Concatenate()([avg_pool, max_pool])

    # Classification head
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="tcn_residual_gesture_classifier")
    return model


def _tcn_residual_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    dilation: int,
    dropout: float,
) -> tf.Tensor:
    """
    Single TCN residual block.

    Structure:
        x ─────────────────────────────┐
        │                              │ (residual/skip connection)
        ├─► Conv1D ─► BN ─► ReLU       │
        │                              │
        ├─► Conv1D ─► BN               │
        │                              │
        └─► Dropout ─► Add ◄───────────┘
                        │
                        └─► ReLU ─► output

    Args:
        x: Input tensor (B, T, C)
        filters: Number of conv filters
        kernel_size: Convolution kernel size
        dilation: Dilation rate for temporal context
        dropout: Dropout rate

    Returns:
        Output tensor with same shape as input
    """
    residual = x

    # First conv: dilated convolution for temporal context
    x = layers.Conv1D(
        filters,
        kernel_size=kernel_size,
        dilation_rate=dilation,
        padding="same",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Second conv: pointwise (1x1) to mix channels
    x = layers.Conv1D(filters, kernel_size=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Dropout before residual add (regularization)
    x = layers.Dropout(dropout)(x)

    # Residual connection: add input to output
    # This allows gradients to flow directly through the skip connection
    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)

    return x



def get_model_summary(config: Config) -> str:
    """
    Get a string summary of the model architecture.

    Args:
        config: Configuration object.

    Returns:
        Model summary string.
    """
    model = build_model(config)
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    return "\n".join(string_list)


def count_parameters(model: keras.Model) -> dict[str, int]:
    """
    Count trainable and non-trainable parameters.

    Args:
        model: Keras model.

    Returns:
        Dictionary with parameter counts.
    """
    trainable = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    non_trainable = sum(
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    )
    return {
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total": trainable + non_trainable,
    }
