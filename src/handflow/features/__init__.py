"""
HandFlow Features Module
=======================

Feature extraction and engineering for hand gesture recognition.
"""

from handflow.features.feature_engineer import (
    FeatureEngineer,
    add_acceleration_features,
    add_velocity_features,
)

__all__ = [
    "FeatureEngineer",
    "add_velocity_features",
    "add_acceleration_features",
]
