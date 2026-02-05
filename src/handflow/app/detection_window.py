# Copyright (c) 2026 Huynh Huy. All rights reserved.

"""
HandFlow Detection Window
========================

Handles the real-time detection loop and camera preview.
Refactored to run UI on main thread (via customtkinter) to avoid macOS threading crashes.
"""

import cv2
import time
import threading
import datetime
import os
import sys
import queue
import numpy as np
import customtkinter as ctk
from PIL import Image
from typing import Optional, Tuple

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

import numpy as np

from handflow.utils import get_logger


# ============================================================
# macOS App Nap Prevention
# ============================================================
# When the app window loses focus, macOS throttles background
# processes ("App Nap"), causing delayed gesture detection.
# This uses multiple approaches to prevent throttling.

_activity_token = None
_caffeinate_process = None

def disable_app_nap():
    """Disable macOS App Nap using strongest available method."""
    global _activity_token, _caffeinate_process
    if sys.platform != 'darwin':
        return

    # Method 1: PyObjC with NSActivityLatencyCritical (strongest flag)
    try:
        from Foundation import NSProcessInfo
        # NSActivityLatencyCritical = 0xFF00000000 (prevents App Nap completely)
        # NSActivityUserInitiated = 0x00FFFFFF (user-initiated, high priority)
        # Combined for maximum effect
        NSActivityLatencyCritical = 0xFF00000000
        NSActivityUserInitiated = 0x00FFFFFF
        activity_options = NSActivityLatencyCritical | NSActivityUserInitiated

        process_info = NSProcessInfo.processInfo()
        _activity_token = process_info.beginActivityWithOptions_reason_(
            activity_options,
            "HandFlow real-time gesture detection"
        )
        print("[Detection] App Nap disabled (PyObjC NSActivityLatencyCritical)")
        return  # Success, no need for fallback
    except ImportError:
        print("[Detection] PyObjC not available, trying caffeinate fallback...")
    except Exception as e:
        print(f"[Detection] PyObjC method failed: {e}, trying caffeinate fallback...")

    # Method 2: Fallback to caffeinate command (reliable but spawns subprocess)
    try:
        import subprocess
        # caffeinate -i: prevent idle sleep (keeps process active)
        # We run it as a subprocess that we'll kill on cleanup
        _caffeinate_process = subprocess.Popen(
            ['caffeinate', '-i', '-w', str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("[Detection] App Nap disabled (caffeinate fallback)")
    except Exception as e:
        print(f"[Detection] WARNING: Could not disable App Nap: {e}")
        print("[Detection] App may be less responsive when window is not focused")


def enable_app_nap():
    """Re-enable App Nap (cleanup)."""
    global _activity_token, _caffeinate_process

    if sys.platform != 'darwin':
        return

    # Clean up PyObjC activity token
    if _activity_token is not None:
        try:
            from Foundation import NSProcessInfo
            process_info = NSProcessInfo.processInfo()
            process_info.endActivity_(_activity_token)
            _activity_token = None
        except:
            pass

    # Clean up caffeinate process
    if _caffeinate_process is not None:
        try:
            _caffeinate_process.terminate()
            _caffeinate_process.wait(timeout=1.0)
            _caffeinate_process = None
        except:
            pass

from handflow.utils import Setting

from handflow.detector import GestureDetector, ArUcoScreenDetector, MacroPadManager
from handflow.actions import ActionExecutor
# Import constant directly to avoid per-frame import overhead
SCREEN_OVERLAY_SET_ID = 20  # From screen_overlay_macropad module


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
        self._screen_overlay_debug = False  # O key to toggle - shows macropad detection info

        # Recording state (R key to toggle) - simple queue-based
        self._recording = False
        self._recording_thread: Optional[threading.Thread] = None
        self._recording_stop_event: Optional[threading.Event] = None
        self._recording_start_time: Optional[float] = None
        self._recording_frame_count = 0
        self._recording_filename: Optional[str] = None
        self._recording_queue: Optional[queue.Queue] = None  # Simple frame queue
        self._recording_writer = None  # VideoWriter in main thread

        # Track last gesture for touch detection optimization
        self._last_gesture = "none"
        self._last_detections = {}
        self._finger_in_detected_area = False  # Track if finger is in ArUco/macropad area
        
        # Screen overlay macropad (just displays markers - detection via macropad_manager)
        self._screen_overlay = None  # Optional[ScreenOverlayMacroPad]
        self._overlay_cmd_lock = threading.Lock()
        self._overlay_cmd_show: Optional[bool] = None  # True=show, False=hide
        self._overlay_cmd_force_hide: bool = False
        self._overlay_cmd_hovered_button: Optional[int] = None  # For hover visual feedback
        self._overlay_cmd_activate: bool = False  # Activate hovered button on touch/touch_hold
        self._overlay_touch_processed: bool = False  # Track if current touch has been processed
        self._overlay_last_activation_time: float = 0.0  # Prevent double activation
        # Auto-hide when finger not in overlay region (after markers detected)
        self._overlay_no_hover_frames: int = 0  # Frames with no finger in region
        self._overlay_no_hover_threshold: int = 5  # Hide after this many frames

        # Window setup (16:9 aspect ratio to match training data)
        self.title("HandFlow v2.0 - Detection Preview [H/V/S/D/C/O/R/Q]")
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

        # Initialize screen overlay macropad if enabled
        if self.setting.screen_overlay_macropad_enabled:
            from handflow.app.screen_overlay_macropad import ScreenOverlayMacroPad
            self._screen_overlay = ScreenOverlayMacroPad(setting, executor)
            self.logger.info("[Detection] Screen overlay macropad initialized")

        # Paper macropad visual feedback overlay
        from handflow.app.paper_macropad_feedback import PaperMacroPadFeedback
        self._paper_feedback = PaperMacroPadFeedback()
        self._paper_feedback_last_hover: Optional[int] = None
        self._paper_feedback_last_activated: Optional[int] = None
        # Track last activation time to detect new activations reliably
        self._paper_feedback_last_activation_time: float = 0.0

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
        self._gesture_model_interval = 1  # Run gesture TCN model every frame (adaptive sampling handles data rate)
        self._aruco_interval = 2  # Run ArUco/MacroPad every N frames
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
        self.bind("<Key-o>", self._toggle_screen_overlay_debug)
        self.bind("<Key-O>", self._toggle_screen_overlay_debug)
        self.bind("<Key-r>", self._toggle_recording)
        self.bind("<Key-R>", self._toggle_recording)
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

    def _toggle_screen_overlay_debug(self, event=None):
        """Toggle screen overlay macropad debug info."""
        self._screen_overlay_debug = not self._screen_overlay_debug
        self._update_status_label()
        print(f"[Detection] Screen Overlay Debug: {'ON' if self._screen_overlay_debug else 'OFF'}")

    def _toggle_recording(self, event=None):
        """Toggle video recording (raw frames at full resolution)."""
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()
        self._update_status_label()

    def _start_recording(self):
        """Start recording - simple queue-based approach."""
        recordings_dir = "recordings"
        os.makedirs(recordings_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._recording_filename = os.path.join(recordings_dir, f"camera_{timestamp}.mp4")

        # Simple bounded queue - drops old frames if writer can't keep up
        self._recording_queue = queue.Queue(maxsize=60)  # ~2 sec buffer at 30fps

        self._recording_stop_event = threading.Event()
        self._recording_start_time = time.time()
        self._recording_frame_count = 0
        self._recording = True

        # Start recording thread
        self._recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
        self._recording_thread.start()

        fps = self._target_fps
        self.logger.info(f"[Recording] Started: {self._recording_filename} @ {fps} FPS")
        print(f"[Recording] Started: {self._recording_filename} @ {fps} FPS")

    def _recording_loop(self):
        """Recording thread - simple queue consumer with lower priority."""
        # Set lower priority for recording thread (UTILITY class = background I/O)
        if sys.platform == 'darwin':
            try:
                import ctypes
                import ctypes.util
                libpthread = ctypes.CDLL(ctypes.util.find_library('pthread'))
                QOS_CLASS_UTILITY = 0x11  # Lower priority for background I/O
                pthread_set_qos = libpthread.pthread_set_qos_class_self_np
                pthread_set_qos.argtypes = [ctypes.c_uint, ctypes.c_int]
                pthread_set_qos.restype = ctypes.c_int
                pthread_set_qos(QOS_CLASS_UTILITY, 0)
            except:
                pass

        fps = self._target_fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self._recording_filename, fourcc, fps, (1280, 720))

        if not writer.isOpened():
            self.logger.error("[Recording] Failed to open video writer")
            return

        frames_written = 0

        while not self._recording_stop_event.is_set():
            try:
                # Block with timeout so we can check stop event
                frame = self._recording_queue.get(timeout=0.1)
                writer.write(frame)
                frames_written += 1
                self._recording_frame_count = frames_written
            except queue.Empty:
                continue

        # Drain remaining frames
        while not self._recording_queue.empty():
            try:
                frame = self._recording_queue.get_nowait()
                writer.write(frame)
                frames_written += 1
                self._recording_frame_count = frames_written
            except queue.Empty:
                break

        writer.release()
        self.logger.info(f"[Recording] Thread done, wrote {frames_written} frames")

    def _stop_recording(self):
        """Stop recording thread."""
        self._recording = False

        if self._recording_stop_event:
            self._recording_stop_event.set()

        if self._recording_thread and self._recording_thread.is_alive():
            self._recording_thread.join(timeout=3.0)

        self._recording_thread = None
        self._recording_stop_event = None
        self._recording_queue = None

        duration = time.time() - self._recording_start_time if self._recording_start_time else 0

        self.logger.info(f"[Recording] Stopped after {duration:.1f}s")
        print(f"\n[Recording] Saved: {self._recording_filename}")
        print(f"  Frames: {self._recording_frame_count}")
        print(f"  Duration: {duration:.1f} seconds")
        if duration > 0:
            print(f"  Actual FPS: {self._recording_frame_count / duration:.1f}")

        self._recording_start_time = None
        self._recording_frame_count = 0

    def _update_status_label(self):
        """Update status bar with current setting state."""
        h_flip = "ON" if self.setting.camera.flip_horizontal else "OFF"
        v_flip = "ON" if self.setting.camera.flip_vertical else "OFF"
        swap = "ON" if self.setting.camera.swap_hands else "OFF"
        draw = "OFF" if self._disable_drawing else "ON"
        cap = "ON" if self._fps_cap_enabled else "OFF"
        overlay_dbg = "ON" if self._screen_overlay_debug else "OFF"
        rec = "REC" if self._recording else "OFF"

        status_text = f"H:{h_flip} V:{v_flip} Swap:{swap} Draw:{draw} Cap:{cap} OvlDbg:{overlay_dbg} Rec:{rec}"
        self.status_label.configure(text=status_text)

        # Visual indicator for recording
        if self._recording:
            self.status_label.configure(text_color="red")
        else:
            self.status_label.configure(text_color=("gray10", "gray90"))

    def start(self):
        """Start detection."""
        if self._running:
            return

        # Disable macOS App Nap for consistent background performance
        disable_app_nap()

        self.logger.info("Starting detection window.")
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        # Start UI update loop
        self._update_ui()
        
    def stop(self):
        """Stop detection and close window."""
        self._running = False

        # Stop recording if active
        if self._recording:
            self._stop_recording()

        if self._thread:
            self._thread.join(timeout=1.0)

        if self._cap:
            self._cap.release()

        # Clean up screen overlay
        if self._screen_overlay:
            self._screen_overlay.hide()
            self._screen_overlay.destroy()

        # Clean up paper macropad feedback
        if self._paper_feedback:
            self._paper_feedback.destroy()

        # Re-enable macOS App Nap
        enable_app_nap()

        self.gesture_Detector.close()
        self.destroy()
        
    def _capture_loop(self):
        """Background thread for CV processing with optimized detection."""
        # Boost thread priority for consistent timing even when backgrounded
        if sys.platform == 'darwin':
            try:
                import ctypes
                import ctypes.util
                # Load libpthread for QoS setting
                libpthread = ctypes.CDLL(ctypes.util.find_library('pthread'))
                # pthread_set_qos_class_self_np(qos_class, relative_priority)
                # QOS_CLASS_USER_INTERACTIVE = 0x21 (highest priority for UI responsiveness)
                QOS_CLASS_USER_INTERACTIVE = 0x21
                pthread_set_qos = libpthread.pthread_set_qos_class_self_np
                pthread_set_qos.argtypes = [ctypes.c_uint, ctypes.c_int]
                pthread_set_qos.restype = ctypes.c_int
                result = pthread_set_qos(QOS_CLASS_USER_INTERACTIVE, 0)
                if result == 0:
                    print("[Detection] Thread QoS set to USER_INTERACTIVE (highest)")
            except Exception as e:
                print(f"[Detection] Could not set thread QoS: {e}")

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
            # Minimize buffer for lowest latency (1 frame = most recent)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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

                    # Queue frame for recording (non-blocking, drops if full)
                    if self._recording and self._recording_queue is not None:
                        try:
                            self._recording_queue.put_nowait(frame.copy())
                        except queue.Full:
                            pass  # Drop frame if queue full - keeps main loop fast

                    self._frame_count += 1

                    # Resize to MediaPipe input size (640x360) - used for both processing AND display
                    # This saves CPU: smaller image for color conversion, PIL, and CTkImage
                    # CTkImage will scale up to window size (blurry but fast)
                    frame_small = cv2.resize(frame, (self._mp_width, self._mp_height), interpolation=cv2.INTER_NEAREST)
                    h_small, w_small = frame_small.shape[:2]

                    # 2. ArUco/MacroPad Detection - every 2 frames
                    run_aruco = (
                        (self._frame_count % self._aruco_interval == 0) or
                        (self._frame_count <= 1)
                    )

                    if run_aruco:
                        # Detect ArUco markers on small frame (same as display)
                        self.aruco_detector.detect(frame_small)

                        # MacroPad uses same markers - pass detected markers directly
                        if self.setting.macropad_enabled:
                            # When screen overlay is visible, prioritize its detection
                            # This prevents paper macropad markers from interfering
                            screen_overlay_visible = (
                                self._screen_overlay is not None and
                                self._screen_overlay.is_visible()
                            )
                            self.macropad_manager.detect_markers(
                                frame_small,
                                prioritize_screen_overlay=screen_overlay_visible
                            )

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

                        # Check macropad area (paper or screen overlay)
                        if self.setting.macropad_enabled and self.macropad_manager.is_detected():
                            if self.macropad_manager._detector.is_point_in_region(pixel_tip):
                                self._finger_in_detected_area = True
                                macropad_active = True  # Finger is in macropad region
                            hovered = self.macropad_manager._detector.get_button_at_point(pixel_tip)
                            if hovered is not None:
                                macropad_active = True

                        # Also check if screen overlay is visible and active
                        if self._screen_overlay and self._screen_overlay.is_visible():
                            macropad_active = True  # Screen overlay handles touch

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
                    current_gesture = "none"
                    for hand in ['Right', 'Left']:
                        if hand in detections and 'gesture' in detections[hand]:
                            current_gesture = detections[hand]['gesture']
                            break
                    self._last_gesture = current_gesture

                    # Get right hand gesture specifically for screen overlay (only right hand triggers overlay)
                    right_hand_gesture = "none"
                    if 'Right' in detections and 'gesture' in detections['Right']:
                        right_hand_gesture = detections['Right']['gesture']

                    # 6. Screen Overlay MacroPad handling (if enabled)
                    # Overlay displays markers, macropad_manager handles detection
                    # Only right hand can trigger screen overlay (left hand gestures ignored)
                    if self._screen_overlay is not None and self.setting.screen_overlay_macropad_enabled:
                        self._handle_screen_overlay(right_hand_gesture)
                    
                    # 7. Draw debug overlays (ArUco and MacroPad) - skip if drawing disabled
                    if not self._disable_drawing:
                        output = self.aruco_detector.draw_debug(output)

                    if self.setting.macropad_enabled:
                        # Update finger state for macropad interaction
                        # Only right hand can interact with macropad (left hand gestures ignored)
                        if 'Right' in detections:
                            info = detections['Right']
                            if 'index_tip' in info:
                                idx_norm = info['index_tip']
                                pixel_tip = (idx_norm[0] * w_small, idx_norm[1] * h_small)
                                gesture = info.get('gesture', 'none')
                                is_touching = gesture == 'touch'
                                # Skip activation when screen overlay is visible - it handles activation separately
                                # This prevents double activation (one from update_finger_state, one from _process_overlay_commands)
                                screen_overlay_active = (
                                    self._screen_overlay is not None and
                                    self._screen_overlay.is_visible()
                                )
                                self.macropad_manager.update_finger_state(
                                    pixel_tip,
                                    is_touching,
                                    skip_activation=screen_overlay_active
                                )

                        if not self._disable_drawing:
                            output = self.macropad_manager.draw_debug(output)

                    # 8. Screen overlay debug info (press 'O' to toggle)
                    if self._screen_overlay_debug and self._screen_overlay:
                        overlay_visible = self._screen_overlay.is_visible()
                        detected_set = self.macropad_manager._detector.current_set_id
                        is_screen_set = detected_set == SCREEN_OVERLAY_SET_ID
                        hovered_btn = self.macropad_manager._hovered_button

                        # Draw debug text
                        debug_y = 50
                        color = (0, 255, 0) if is_screen_set else (0, 165, 255)
                        cv2.putText(output, f"[ScreenOverlay Debug]", (10, debug_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        debug_y += 20
                        cv2.putText(output, f"Overlay visible: {overlay_visible}", (10, debug_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        debug_y += 20
                        cv2.putText(output, f"Detected set: {detected_set} (screen={SCREEN_OVERLAY_SET_ID})", (10, debug_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        debug_y += 20
                        cv2.putText(output, f"MacroPad valid: {self.macropad_manager.is_detected()}", (10, debug_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        debug_y += 20
                        hover_color = (0, 255, 0) if hovered_btn is not None else (128, 128, 128)
                        cv2.putText(output, f"Hovered button: {hovered_btn}", (10, debug_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, hover_color, 1)

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

        # Process screen overlay commands (must run on main thread)
        if self._screen_overlay:
            if self.setting.screen_overlay_macropad_enabled:
                self._process_overlay_commands()
            else:
                # Hide overlay if setting was disabled
                if self._screen_overlay.is_visible():
                    self._screen_overlay.hide()

        # Process paper macropad feedback (only when screen overlay is NOT visible)
        self._process_paper_feedback()

        # Schedule next update (~24 FPS - sufficient for preview, saves CPU)
        if self._running:
            self.after(42, self._update_ui)

    def _process_overlay_commands(self):
        """
        Process overlay commands from background thread (runs on main thread).

        The screen overlay ONLY displays markers - all detection is handled by
        macropad_manager which sees the markers through the camera.
        """
        with self._overlay_cmd_lock:
            if self._overlay_cmd_show is True:
                if not self._screen_overlay.is_visible():
                    if not self._screen_overlay.is_in_cooldown():
                        self._screen_overlay.show()
                else:
                    self._screen_overlay.reset_hide_grace()

                # Update hover visual on overlay (feedback from macropad_manager detection)
                self._screen_overlay.set_hovered_button(self._overlay_cmd_hovered_button)

                # Activate button on touch/touch_hold gesture
                print(f"[ScreenOverlay] Checking activation: activate={self._overlay_cmd_activate}, hovered={self._overlay_cmd_hovered_button}")
                if self._overlay_cmd_activate and self._overlay_cmd_hovered_button is not None:
                    # Check time-based cooldown to prevent double activation
                    current_time = time.time()
                    if current_time - self._overlay_last_activation_time < 0.5:
                        print(f"[ScreenOverlay] Skipping - too soon since last activation ({current_time - self._overlay_last_activation_time:.2f}s)")
                        self._overlay_cmd_activate = False
                    else:
                        # Execute button action via macropad_manager
                        # IMPORTANT: Force set ID to screen overlay (20) to ensure correct button set is used
                        print(f"[ScreenOverlay] >>> ACTIVATING BUTTON {self._overlay_cmd_hovered_button} from SET {SCREEN_OVERLAY_SET_ID} <<<")
                        self.macropad_manager._activate_button(
                            self._overlay_cmd_hovered_button,
                            force_set_id=SCREEN_OVERLAY_SET_ID
                        )
                        self._overlay_last_activation_time = current_time
                        self.logger.info(f"[ScreenOverlay] Activated button {self._overlay_cmd_hovered_button} from set {SCREEN_OVERLAY_SET_ID}")
                        self._overlay_cmd_activate = False

                    # Hide overlay after activation and set cooldown
                    if self._overlay_cmd_force_hide:
                        self._screen_overlay.hide()
                        self._screen_overlay.set_cooldown(1.0)  # 1 second cooldown before touch_hover can show again
                        self._overlay_cmd_force_hide = False
                        print(f"[ScreenOverlay] Hidden with 1.0s cooldown")

                elif self._overlay_cmd_activate and self._overlay_cmd_hovered_button is None:
                    print(f"[ScreenOverlay] Activation requested but no hovered button!")
                    # Still hide and cooldown even if no button was hovered
                    if self._overlay_cmd_force_hide:
                        self._screen_overlay.hide()
                        self._screen_overlay.set_cooldown(0.5)
                        self._overlay_cmd_force_hide = False

            elif self._overlay_cmd_show is False:
                if self._screen_overlay.is_visible():
                    if self._overlay_cmd_force_hide:
                        self._screen_overlay.hide()
                        self._screen_overlay.set_cooldown(0.7)
                        self._overlay_cmd_force_hide = False
                    else:
                        self._screen_overlay.request_hide()

            self._overlay_cmd_show = None
            self._overlay_cmd_activate = False

        self._screen_overlay.update()

    def _process_paper_feedback(self):
        """
        Process paper macropad visual feedback (runs on main thread).

        Shows hover/click feedback overlay for paper macropad interactions.
        Only active when screen overlay is NOT visible.
        """
        # Skip if screen overlay is visible (it has its own feedback)
        if self._screen_overlay and self._screen_overlay.is_visible():
            # Hide paper feedback if it was showing
            if self._paper_feedback_last_hover is not None:
                self._paper_feedback.set_hovered_button(None)
                self._paper_feedback_last_hover = None
            return

        # Skip if macropad not enabled or not detected
        if not self.setting.macropad_enabled or not self.macropad_manager.is_detected():
            if self._paper_feedback_last_hover is not None:
                self._paper_feedback.set_hovered_button(None)
                self._paper_feedback_last_hover = None
            return

        # Skip if screen overlay set is detected (ID 20)
        detected_set = self.macropad_manager._detector.current_set_id
        if detected_set == SCREEN_OVERLAY_SET_ID:
            return

        # Update button labels from active set
        active_set = self.macropad_manager.active_set
        if active_set:
            labels = [
                active_set.buttons.get(i).label if active_set.buttons.get(i) and active_set.buttons.get(i).label else f"Button {i+1}"
                for i in range(12)
            ]
            self._paper_feedback.set_button_labels(labels)

        # Check current hover state
        current_hover = self.macropad_manager._hovered_button

        # Update hover feedback if changed
        if current_hover != self._paper_feedback_last_hover:
            self._paper_feedback.set_hovered_button(current_hover)
            self._paper_feedback_last_hover = current_hover

        # Check for activation using activation log (more reliable than _activated_button)
        # The log has timestamps so we can detect new activations even if _activated_button was cleared
        activation_log = self.macropad_manager._activation_log
        if activation_log:
            latest_activation = activation_log[-1]
            # Check if this is a new activation (timestamp changed)
            if latest_activation.timestamp > self._paper_feedback_last_activation_time:
                self._paper_feedback.show_click_feedback(latest_activation.button_idx)
                self._paper_feedback_last_activation_time = latest_activation.timestamp

        # Process Tk events
        self._paper_feedback.update()

    def _is_point_in_expanded_region(self, point: Tuple[float, float], region: np.ndarray, margin_ratio: float = 0.25) -> bool:
        """
        Check if a point is within an expanded version of the detection region.

        Args:
            point: (x, y) point to check
            region: 4 corners of the detection region [TL, TR, BR, BL]
            margin_ratio: How much to expand (0.25 = expand by 25% of region size on each side)

        Returns:
            True if point is within the expanded region
        """
        if region is None or len(region) != 4:
            return False

        # Calculate region center
        center = np.mean(region, axis=0)

        # Expand each corner outward from center
        expanded = []
        for corner in region:
            direction = corner - center
            expanded_corner = corner + direction * margin_ratio
            expanded.append(expanded_corner)

        expanded_region = np.array(expanded, dtype=np.float32)

        # Check if point is inside expanded polygon
        result = cv2.pointPolygonTest(expanded_region, point, False)
        return result >= 0

    def _handle_screen_overlay(self, current_gesture: str):
        """
        Handle screen overlay show/hide based on gesture.

        The overlay ONLY displays ArUco markers - same as paper macropad.
        Detection is handled by macropad_manager which sees the markers via camera.

        Gestures:
        - touch_hover: Show overlay (only if finger NOT hovering over paper macropad)
        - touch / touch_hold: Activate if screen overlay detected, then hide
        - other: Hide overlay
        """
        # Check what's currently detected
        is_detected = self.macropad_manager.is_detected()
        detected_set = self.macropad_manager._detector.current_set_id if is_detected else None
        mp_hovered = self.macropad_manager._hovered_button

        # Check if finger is near paper macropad region (set IDs 12, 13, 14)
        # Use expanded region (with margin) to suppress screen overlay even when
        # finger is near but not exactly over a button (e.g., due to projection angle)
        finger_over_paper_macropad = False
        if is_detected and detected_set in (12, 13, 14):
            finger_pos = self.macropad_manager._finger_pos
            detection = self.macropad_manager._detector.detection
            if finger_pos and detection is not None:
                # Check if finger is within expanded region (original + ~1 marker size margin)
                # margin_ratio=0.25 expands by 25% on each side, roughly 1 marker size
                finger_over_paper_macropad = self._is_point_in_expanded_region(
                    finger_pos, detection.detection_region, margin_ratio=0.25
                )

        # Get hovered button only if screen overlay is detected
        hovered_btn = None
        if is_detected and detected_set == SCREEN_OVERLAY_SET_ID:
            hovered_btn = mp_hovered

        with self._overlay_cmd_lock:
            self._overlay_cmd_hovered_button = hovered_btn

            if current_gesture == 'touch_hover':
                # Reset touch processed flag when not touching
                self._overlay_touch_processed = False
                # Only show overlay if finger is NOT over paper macropad
                if not finger_over_paper_macropad:
                    self._overlay_cmd_show = True
                    self._overlay_cmd_activate = False
                    self._overlay_cmd_force_hide = False

                    # Auto-hide check: if overlay is visible AND screen overlay markers detected
                    # but finger is not over any button, count frames and hide after threshold
                    if (self._screen_overlay and self._screen_overlay.is_visible() and
                        is_detected and detected_set == SCREEN_OVERLAY_SET_ID):
                        # Screen overlay markers are detected by camera
                        if hovered_btn is None:
                            # Finger not over any button - increment counter
                            self._overlay_no_hover_frames += 1
                            if self._overlay_no_hover_frames >= self._overlay_no_hover_threshold:
                                # Hide overlay - finger not in region for too long
                                print(f"[ScreenOverlay] Auto-hide: no finger in region for {self._overlay_no_hover_frames} frames")
                                self._overlay_cmd_show = False
                                self._overlay_cmd_force_hide = True
                                self._overlay_no_hover_frames = 0
                        else:
                            # Finger is over a button - reset counter
                            self._overlay_no_hover_frames = 0
                    elif not (is_detected and detected_set == SCREEN_OVERLAY_SET_ID):
                        # Markers not detected yet - don't count (camera still acquiring)
                        pass
                else:
                    # Finger is over paper macropad - don't show screen overlay
                    self._overlay_cmd_show = False
                    self._overlay_cmd_activate = False
                    self._overlay_cmd_force_hide = False
                    self._overlay_no_hover_frames = 0  # Reset counter

            elif current_gesture in ('touch', 'touch_hold'):
                # Reset no-hover counter on touch
                self._overlay_no_hover_frames = 0
                # Only activate screen overlay if finger is NOT over paper macropad
                # AND we haven't already processed this touch
                if not finger_over_paper_macropad and not self._overlay_touch_processed:
                    self._overlay_cmd_show = True
                    self._overlay_cmd_activate = True
                    self._overlay_cmd_force_hide = True
                    self._overlay_touch_processed = True  # Mark as processed
                    print(f"[ScreenOverlay] TOUCH detected! gesture={current_gesture}, hovered_btn={hovered_btn}")
                elif finger_over_paper_macropad:
                    # Finger is over paper macropad - don't show/activate screen overlay
                    self._overlay_cmd_show = False
                    self._overlay_cmd_activate = False
                    self._overlay_cmd_force_hide = False

            else:
                # No relevant gesture - hide overlay and reset touch flag
                self._overlay_touch_processed = False
                self._overlay_cmd_show = False
                self._overlay_cmd_activate = False
                self._overlay_cmd_force_hide = False
                self._overlay_no_hover_frames = 0  # Reset counter
