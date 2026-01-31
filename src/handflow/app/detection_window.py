"""
HandFlow Detection Window
========================

Handles the real-time detection loop and camera preview.
Refactored to run UI on main thread (via customtkinter) to avoid macOS threading crashes.
"""

import cv2
import time
import threading
import customtkinter as ctk
from PIL import Image
from typing import Optional

from handflow.utils import get_logger

from handflow.utils import Setting

from handflow.detector import GestureDetector, ArUcoScreenDetector, MacroPadManager
from handflow.actions import ActionExecutor


class DetectionWindow(ctk.CTkToplevel):
    """
    Detection window showing camera preview and debug info.
    
    Architecture:
    - Main Thread: Handles UI updates (customtkinter)
    - Background Thread: Handles Camera capture & Computer Vision processing
    """
    
    def __init__(self, setting: Setting, executor: ActionExecutor):
        super().__init__()

        self.logger = get_logger("handflow.detection_window")

        self.setting = setting
        self.executor = executor

        # Window setup
        self.title("HandFlow v2.0 - Detection Preview [H=Flip H | V=Flip V | S=Swap L/R | Q=Quit]")
        self.geometry("1280x720")

        # UI Elements
        self.video_label = ctk.CTkLabel(self, text="Starting camera...")
        self.video_label.pack(fill="both", expand=True)

        # Status bar for flip info
        self.status_frame = ctk.CTkFrame(self, height=30)
        self.status_frame.pack(fill="x", side="bottom")
        self.status_label = ctk.CTkLabel(self.status_frame, text="", font=ctk.CTkFont(size=12))
        self.status_label.pack(side="left", padx=10)
        self._update_status_label()

        # CV Components
        # Get screen size for ArUco detector
        sw, sh = 1920, 1080
        try:
            import pyautogui
            sw, sh = pyautogui.size()
        except:
            pass

        # Create ArUco detector first, then pass to GestureDetector
        self.aruco_detector = ArUcoScreenDetector(screen_width=sw, screen_height=sh)

        # Pass ArUco detector to GestureDetector for touch-to-cursor mapping
        self.gesture_Detector = GestureDetector(setting, executor, aruco_detector=self.aruco_detector)

        # detection_mode: "balanced" or "motion_priority"
        self.macropad_manager = MacroPadManager(setting, executor, detection_mode="balanced")

        # State
        self._running = False
        self._thread = None
        self._cap = None
        self._latest_frame = None
        self._lock = threading.Lock()

        # Handle close
        self.protocol("WM_DELETE_WINDOW", self.stop)

        # Keyboard shortcuts
        self.bind("<Key-h>", self._toggle_flip_horizontal)
        self.bind("<Key-H>", self._toggle_flip_horizontal)
        self.bind("<Key-v>", self._toggle_flip_vertical)
        self.bind("<Key-V>", self._toggle_flip_vertical)
        self.bind("<Key-s>", self._toggle_swap_hands)
        self.bind("<Key-S>", self._toggle_swap_hands)
        self.bind("<Key-q>", lambda e: self.stop())
        self.bind("<Key-Q>", lambda e: self.stop())
        self.bind("<Escape>", lambda e: self.stop())

        # Focus the window to receive key events
        self.focus_force()
        
    def _toggle_flip_horizontal(self, event=None):
        """Toggle horizontal flip."""
        self.setting.camera.flip_horizontal = not self.setting.camera.flip_horizontal
        self._update_status_label()
        print(f"[Detection] Horizontal flip: {'ON' if self.setting.camera.flip_horizontal else 'OFF'}")

    def _toggle_flip_vertical(self, event=None):
        """Toggle vertical flip."""
        self.setting.camera.flip_vertical = not self.setting.camera.flip_vertical
        self._update_status_label()
        print(f"[Detection] Vertical flip: {'ON' if self.setting.camera.flip_vertical else 'OFF'}")

    def _toggle_swap_hands(self, event=None):
        """Toggle swap hands (L/R labels)."""
        self.setting.camera.swap_hands = not self.setting.camera.swap_hands
        self._update_status_label()
        print(f"[Detection] Swap hands: {'ON' if self.setting.camera.swap_hands else 'OFF'}")

    def _update_status_label(self):
        """Update status bar with current setting state."""
        h_flip = "ON" if self.setting.camera.flip_horizontal else "OFF"
        v_flip = "ON" if self.setting.camera.flip_vertical else "OFF"
        swap = "ON" if self.setting.camera.swap_hands else "OFF"
        self.status_label.configure(text=f"Flip H: {h_flip} | Flip V: {v_flip} | Swap L/R: {swap} | Keys: H/V/S/Q")

    def start(self):
        """Start detection."""
        if self._running:
            return
     
        self.logger.info("Starting detection window.")
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        # Start UI update loop
        self._update_ui()
        
    def stop(self):
        """Stop detection and close window."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        
        if self._cap:
            self._cap.release()
            
        self.gesture_Detector.close()
        self.destroy()
        
    def _capture_loop(self):
        """Background thread for CV processing."""
        cam_idx = self.setting.camera.index
        self._cap = cv2.VideoCapture(cam_idx)
        
        # Optimize camera
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not self._cap.isOpened():
            self.logger.info(f"[Detection] Error: Could not open camera {cam_idx}")
            self._running = False
            return
            
        self.logger.info("[Detection] Processing started.")
        
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            # 1. Preprocessing
            if self.setting.camera.flip_horizontal:
                frame = cv2.flip(frame, 1)
            if self.setting.camera.flip_vertical:
                frame = cv2.flip(frame, 0)
                
            output = frame.copy()
            h, w, _ = frame.shape

            # 2. ArUco Screen Detection
            self.aruco_detector.detect(frame)
            output = self.aruco_detector.draw_debug(output)

            # 3. Macro Pad Detection (disable cursor movements when finger is over macropad
            macropad_active = False
            if self.setting.macropad_enabled:
                self.macropad_manager.detect_markers(frame)

                # Check if finger is over macropad using LAST frame's position for stable detection
                if self.macropad_manager.is_detected():
                    last_tip = self.gesture_Detector._right_index_tip
                    if last_tip is not None:
                        pixel_tip = (last_tip[0] * w, last_tip[1] * h)
                        # Check if finger is over any button
                        hovered = self.macropad_manager._detector.get_button_at_point(pixel_tip)
                        if hovered is not None:
                            macropad_active = True

            # 4. Tell gesture Detector whether macropad is handling interaction this prevents cursor movements from interfering with macropad
            self.gesture_Detector.set_macropad_active(macropad_active)

            # 5. Gesture Recognition (calling from gesture Detector)
            output, detections = self.gesture_Detector.process_frame(output)

            # 6. Integrate Gesture Tip with Macro Pad
            if self.setting.macropad_enabled:
                primary_hand = 'Right' if 'Right' in detections else 'Left'
                if primary_hand in detections:
                    info = detections[primary_hand]
                    if 'index_tip' in info:
                        idx_norm = info['index_tip']
                        pixel_tip = (idx_norm[0] * w, idx_norm[1] * h)

                        gesture = info.get('gesture', 'none')
                        is_touching = gesture == 'touch'

                        # Debug: show gesture when over macropad
                        if self.macropad_manager.is_detected() and self.macropad_manager._hovered_button is not None:
                            self.logger.info(f"[Detection] Gesture={gesture}, is_touching={is_touching}, hovered={self.macropad_manager._hovered_button}")

                        self.macropad_manager.update_finger_state(pixel_tip, is_touching)

                output = self.macropad_manager.draw_debug(output)
            
            # Update latest frame safely
            with self._lock:
                self._latest_frame = output
                
    def _update_ui(self):
        """Main thread UI update loop."""
        if not self._running:
            return

        image = None
        with self._lock:
            if self._latest_frame is not None:
                # Convert BGR (OpenCV) to RGB (PIL)
                rgb = cv2.cvtColor(self._latest_frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
        
        if image:
            # Create/Update CTkImage
            # Resize logic if needed, or let CTkImage handle it
            ctk_img = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
            self.video_label.configure(image=ctk_img, text="")
        
        # Schedule next update (30 FPS approx)
        self.after(33, self._update_ui)
