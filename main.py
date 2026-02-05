#!/usr/bin/env python3
# Copyright (c) 2026 Huynh Huy. All rights reserved.

"""
HandFlow - Gesture & Macro Pad Control System
=============================================

Professional hand gesture recognition and paper macro pad control system.
Uses ArUco markers for screen boundary detection and programmable touch buttons.

Main entry point for the application.

Usage:
    python main.py              # Launch GUI application
    python main.py --cli        # Launch in CLI mode (detection only)
    python main.py --calibrate  # Launch calibration mode
"""

import sys
import os
import argparse

from handflow.utils.logging import setup_logging, get_logger

# Ensure src is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Launch the main GUI application."""
    #set up logging
    log_file = "logs/app.log"
    setup_logging(level="INFO", log_file=log_file)
    logger = get_logger("handflow.main")

    try:
        import customtkinter as ctk
        from handflow.app import HandFlowApp

        # Configure appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        app = HandFlowApp()
        app.mainloop()

    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
