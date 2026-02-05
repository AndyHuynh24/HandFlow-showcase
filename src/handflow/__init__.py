# Copyright (c) 2026 Huynh Huy. All rights reserved.

"""
HandFlow - Gesture-Controlled Computer Interaction
==================================================

A machine-learning powered application that lets you control your computer
using hand gestures and movement.

Features:
    - 12 well-trained gesture recognition
    - Custom action mapping (shortcuts, mouse control, etc.)
    - Real-time detection with MediaPipe
    - Cross-platform support (macOS, Windows, Linux)

Example:
    >>> from handflow import HandFlowApp
    >>> app = HandFlowApp()
    >>> app.run()

"""

__version__ = "1.0.0"
__author__ = "HandFlow Team"

from handflow.utils.config import Config, load_config
from handflow.utils.setting import Setting, load_setting, save_setting
from handflow.utils.logging import setup_logging

__all__ = [
    "__version__",
    "Config",
    "load_config",
    "Setting", 
    "load_setting", 
    "save_setting",
    "setup_logging",
]
