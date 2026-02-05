# Copyright (c) 2026 Huynh Huy. All rights reserved.

import os
import yaml
import sys
import shutil
import unittest
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from handflow.detector.screen_detector import ArUcoScreenDetector, ArUcoCalibration

class TestCalibrationMigration(unittest.TestCase):
    def setUp(self):
        self.config_dir = Path("config_test")
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / "aruco_calibration.yaml"
        
        # Original values from user's file
        self.initial_data = {
            "top_left": {"horizontal": 85.0, "vertical": -101.0},
            "top_right": {"horizontal": 99.0, "vertical": -104.0},
            "bottom_right": {"horizontal": 100.0, "vertical": -18.0},
            "bottom_left": {"horizontal": 86.0, "vertical": -36.0}
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(self.initial_data, f)

    def tearDown(self):
        if self.config_dir.exists():
            shutil.rmtree(self.config_dir)

    def test_load_calibration(self):
        detector = ArUcoScreenDetector(config_file=str(self.config_file))
        
        # Verify values loaded correctly
        self.assertEqual(detector.calibration.top_left.horizontal, 85.0)
        self.assertEqual(detector.calibration.top_left.vertical, -101.0)
        self.assertEqual(detector.calibration.top_right.horizontal, 99.0)
        
        print("Successfully loaded calibration from YAML")

    def test_save_calibration(self):
        detector = ArUcoScreenDetector(config_file=str(self.config_file))
        
        # Modify values
        detector.calibration.top_left.horizontal = 90.5
        detector.calibration.top_left.vertical = -105.2
        
        # Save
        detector.save_calibration()
        
        # Verify file content
        with open(self.config_file, 'r') as f:
            saved_data = yaml.safe_load(f)
            
        self.assertEqual(saved_data['top_left']['horizontal'], 90.5)
        self.assertEqual(saved_data['top_left']['vertical'], -105.2)
        
        print("Successfully saved calibration to YAML")

if __name__ == '__main__':
    unittest.main()
