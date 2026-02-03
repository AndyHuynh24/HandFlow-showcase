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

        # Debug toggles (can be toggled with keyboard) - must be defined before _update_status_label
        self._disable_drawing = True  # D key to toggle
        self._fps_cap_enabled = True  # C key to toggle

        # Track last gesture for touch detection optimization
        self._last_gesture = "none"
        self._last_detections = {}
        self._finger_in_detected_area = False  # Track if finger is in ArUco/macropad area

        # Window setup (16:9 aspect ratio to match training data)
        self.title("HandFlow v2.0 - Detection Preview [H=Flip H | V=Flip V | S=Swap | D=Draw | C=Cap | Q=Quit]")
        self.geometry("1280x750")  # 720 + status bar

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

        # FPS control (read from config for consistency with data collection)
        from handflow.utils import load_config
        config = load_config("config/config.yaml")
        self._target_fps = getattr(config.data, 'target_fps', 20.0)
        self._frame_duration = 1.0 / self._target_fps
        self._gesture_model_interval = 2  # Run gesture TCN model every N frames
        self._aruco_interval = 3  # Run ArUco/MacroPad every N frames
        self._frame_count = 0
        self._last_detection_results = None  # Cache detection results

        # Resolution for MediaPipe - use 16:9 to match training data (1280x720)
        # 640x360 maintains aspect ratio while being lower resolution for speed
        self._mp_width = 640
        self._mp_height = 360

        # Pre-allocated buffer for RGB conversion (avoid allocation each frame)
        self._rgb_buffer = None

        # Handle close
        self.protocol("WM_DELETE_WINDOW", self.stop)

        # Keyboard shortcuts
        self.bind("<Key-h>", self._toggle_flip_horizontal)
        self.bind("<Key-H>", self._toggle_flip_horizontal)
        self.bind("<Key-v>", self._toggle_flip_vertical)
        self.bind("<Key-V>", self._toggle_flip_vertical)
        self.bind("<Key-s>", self._toggle_swap_hands)
        self.bind("<Key-S>", self._toggle_swap_hands)
        self.bind("<Key-d>", self._toggle_drawing)
        self.bind("<Key-D>", self._toggle_drawing)
        self.bind("<Key-c>", self._toggle_fps_cap)
        self.bind("<Key-C>", self._toggle_fps_cap)
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

    def _toggle_drawing(self, event=None):
        """Toggle debug drawing."""
        self._disable_drawing = not self._disable_drawing
        self._update_status_label()
        print(f"[Detection] Drawing: {'OFF' if self._disable_drawing else 'ON'}")

    def _toggle_fps_cap(self, event=None):
        """Toggle data collection rate limiting (20 FPS)."""
        self._fps_cap_enabled = not self._fps_cap_enabled
        self.gesture_Detector.set_data_rate_limit(self._fps_cap_enabled)
        self._update_status_label()
        print(f"[Detection] Data Rate Limit: {'ON (20 FPS)' if self._fps_cap_enabled else 'OFF (unlimited)'}")

    def _update_status_label(self):
        """Update status bar with current setting state."""
        h_flip = "ON" if self.setting.camera.flip_horizontal else "OFF"
        v_flip = "ON" if self.setting.camera.flip_vertical else "OFF"
        swap = "ON" if self.setting.camera.swap_hands else "OFF"
        draw = "OFF" if self._disable_drawing else "ON"
        cap = "ON" if self._fps_cap_enabled else "OFF"
        self.status_label.configure(text=f"Flip H: {h_flip} | Flip V: {v_flip} | Swap: {swap} | Draw: {draw} | Cap: {cap} | Keys: H/V/S/D/C/Q")

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
        """Background thread for CV processing with optimized detection."""
        cam_idx = self.setting.camera.index
        consecutive_failures = 0
        max_failures = 30  # Stop after 30 consecutive frame read failures

        try:
            self._cap = cv2.VideoCapture(cam_idx)

            # Optimize camera - use 16:9 aspect ratio to match training data
            # 1280x720 matches data collection, then resize to 640x360 for MediaPipe
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._cap.set(cv2.CAP_PROP_FPS, 30)

            if not self._cap.isOpened():
                self.logger.error(f"[Detection] Error: Could not open camera {cam_idx}")
                self._running = False
                return

            self.logger.info(f"[Detection] Processing started. Target: {self._target_fps} FPS, "
                            f"Gesture model every {self._gesture_model_interval} frames, "
                            f"ArUco every {self._aruco_interval} frames")

            last_frame_time = time.time()

            while self._running:
                try:
                    ret, frame = self._cap.read()
                    if not ret:
                        consecutive_failures += 1
                        if consecutive_failures >= max_failures:
                            self.logger.error(f"[Detection] Camera read failed {max_failures} times consecutively. Stopping.")
                            self._running = False
                            break
                        time.sleep(0.01)
                        continue

                    # Reset failure counter on successful read
                    consecutive_failures = 0

                    # Calculate delta time for FPS-invariant features
                    current_time = time.time()
                    delta_time = current_time - last_frame_time
                    last_frame_time = current_time

                    # 1. Preprocessing (in-place flips, no copy needed yet)
                    if self.setting.camera.flip_horizontal:
                        frame = cv2.flip(frame, 1)
                    if self.setting.camera.flip_vertical:
                        frame = cv2.flip(frame, 0)

                    h, w = frame.shape[:2]
                    self._frame_count += 1

                    # Resize to MediaPipe input size (640x360) - used for both processing AND display
                    # This saves CPU: smaller image for color conversion, PIL, and CTkImage
                    # CTkImage will scale up to window size (blurry but fast)
                    frame_small = cv2.resize(frame, (self._mp_width, self._mp_height), interpolation=cv2.INTER_NEAREST)
                    h_small, w_small = frame_small.shape[:2]

                    # 2. ArUco/MacroPad Detection
                    # Run every frame if finger is in detected area (for accurate interaction)
                    # Otherwise run every N frames (markers are static, saves CPU)
                    run_aruco = (
                        self._finger_in_detected_area or  # Every frame when finger is in area
                        (self._frame_count % self._aruco_interval == 0) or
                        (self._frame_count <= 1)
                    )

                    if run_aruco:
                        # Detect ArUco markers on small frame (same as display)
                        self.aruco_detector.detect(frame_small)

                        # MacroPad uses same markers - pass detected markers directly
                        if self.setting.macropad_enabled:
                            self.macropad_manager.detect_markers(frame_small)

                    # 3. Check if finger is in detected area (ArUco screen or macropad)
                    # Used to run ArUco detection every frame when interacting
                    self._finger_in_detected_area = False
                    macropad_active = False
                    last_tip = self.gesture_Detector._right_index_tip
                    if last_tip is not None:
                        pixel_tip = (last_tip[0] * w_small, last_tip[1] * h_small)

                        # Check ArUco screen area
                        if self.aruco_detector.is_point_in_screen(pixel_tip):
                            self._finger_in_detected_area = True

                        # Check macropad area
                        if self.setting.macropad_enabled and self.macropad_manager.is_detected():
                            if self.macropad_manager._detector.is_point_in_region(pixel_tip):
                                self._finger_in_detected_area = True
                            hovered = self.macropad_manager._detector.get_button_at_point(pixel_tip)
                            if hovered is not None:
                                macropad_active = True

                    # 4. Tell gesture detector whether macropad is handling interaction
                    self.gesture_Detector.set_macropad_active(macropad_active)

                    # 5. Gesture Recognition
                    # - MediaPipe runs EVERY frame (for smooth landmarks & proper data collection)
                    # - Gesture TCN model runs every N frames (inference optimization)
                    run_gesture_model = (self._frame_count % self._gesture_model_interval == 0)

                    # Pass delta_time for FPS-invariant velocity features
                    # process_frame draws on frame_small, which is also used for display
                    output, detections = self.gesture_Detector.process_frame(
                        frame_small,  # Small res for both display and MediaPipe
                        frame_small=None,  # Already small, no separate resize needed
                        run_gesture_model=run_gesture_model,  # TCN inference every N frames
                        delta_time=delta_time,
                        disable_drawing=self._disable_drawing  # Debug flag
                    )

                    # Track last gesture for touch optimization
                    self._last_detections = detections
                    for hand in ['Right', 'Left']:
                        if hand in detections and 'gesture' in detections[hand]:
                            self._last_gesture = detections[hand]['gesture']
                            break

                    # 6. Draw debug overlays (ArUco and MacroPad) - skip if drawing disabled
                    if not self._disable_drawing:
                        output = self.aruco_detector.draw_debug(output)

                    if self.setting.macropad_enabled:
                        # Update finger state for macropad interaction
                        primary_hand = 'Right' if 'Right' in detections else 'Left'
                        if primary_hand in detections:
                            info = detections[primary_hand]
                            if 'index_tip' in info:
                                idx_norm = info['index_tip']
                                pixel_tip = (idx_norm[0] * w_small, idx_norm[1] * h_small)
                                gesture = info.get('gesture', 'none')
                                is_touching = gesture == 'touch'
                                self.macropad_manager.update_finger_state(pixel_tip, is_touching)

                        if not self._disable_drawing:
                            output = self.macropad_manager.draw_debug(output)

                    # Update latest frame safely (small frame, will be scaled up by CTkImage)
                    with self._lock:
                        self._latest_frame = output

                except Exception as e:
                    self.logger.error(f"[Detection] Error in frame processing: {e}", exc_info=True)
                    # Continue processing - don't crash on single frame errors
                    continue

        except Exception as e:
            self.logger.error(f"[Detection] Fatal error in capture loop: {e}", exc_info=True)
        finally:
            self.logger.info("[Detection] Capture loop ended.")
            self._running = False

                
    def _update_ui(self):
        """Main thread UI update loop."""
        if not self._running:
            self.logger.info("[Detection] UI update stopped - capture not running.")
            return

        try:
            image = None
            with self._lock:
                if self._latest_frame is not None:
                    # Convert BGR (OpenCV) to RGB (PIL)
                    rgb = cv2.cvtColor(self._latest_frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(rgb)

            if image:
                # Create/Update CTkImage - scale up small frame (640x360) to display size (1280x720)
                # This is faster than processing full resolution frames
                display_size = (1280, 720)
                ctk_img = ctk.CTkImage(light_image=image, dark_image=image, size=display_size)
                self.video_label.configure(image=ctk_img, text="")

        except Exception as e:
            self.logger.error(f"[Detection] Error in UI update: {e}")

        # Schedule next update (~24 FPS - sufficient for preview, saves CPU)
        if self._running:
            self.after(42, self._update_ui)
