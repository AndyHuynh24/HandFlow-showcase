"""
HandFlow Gesture Detector
======================

Core Detector for gesture recognition and processing.
"""

import os
import time
import cv2
import numpy as np
import tensorflow as tf
from typing import Optional, Dict, Tuple, TYPE_CHECKING
from collections import deque, Counter

from handflow.utils import Setting, get_logger
from handflow.utils.smoothing import OneEuroFilter
from handflow.detector.handedness_tracker import HandTracker
import mediapipe as mp



from handflow.actions import ActionExecutor
from handflow.features import FeatureEngineer

if TYPE_CHECKING:
    from handflow.detector import ArUcoScreenDetector


class GestureDetector:
    """
    Core Detector for hand gesture recognition.

    Pipeline:
    1. MediaPipe Hand Tracking -> Raw Landmarks
    2. FeatureEngineer -> Features (positions + distances)
    3. TFLite Model -> Gesture Classification
    4. ActionExecutor -> Trigger Mapped Action
    5. Touch gesture -> Move cursor via ArUco homography
    """
    def __init__(
        self,
        setting: Setting,
        executor: ActionExecutor,
        aruco_detector: Optional["ArUcoScreenDetector"] = None,
    ):
        self.logger = get_logger("handflow.GestureDetector")

        self.setting = setting
        self.executor = executor

        # Load config
        from handflow.utils import load_config
        self.config = load_config("config/config.yaml")

        self.DEFAULT_MODEL = self.config.model.model_path 

        # Get gesture classes from config (same as used in training)
        self.gesture_classes = self.config.model.gestures

        self.logger.info(f"[GestureDetector] Hand gestures: {self.gesture_classes}")

        # Initialize MediaPipe directly 
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=self.config.mediapipe.min_detection_confidence,
            min_tracking_confidence=self.config.mediapipe.min_tracking_confidence,
            max_num_hands=self.config.mediapipe.max_num_hands, 
            model_complexity = self.config.mediapipe.model_complexity
        )

        # Feature engineer
        self.feature_engineer = FeatureEngineer()

        # Models
        self.model_path = self.DEFAULT_MODEL
        self._load_model()

        # Sequence tracking 
        self.sequence_length = self.config.data.sequence_length
        self.right_sequence = []
        self.left_sequence = []
        self.right_lock = False
        self.left_lock = False

        # Prediction history for smoothing
        self.right_predictions = deque(maxlen=10)
        self.left_predictions = deque(maxlen=10)

        # Cooldown 
        self.COOLDOWN_FRAMES = setting.inference.cooldown_frames
        self.right_cooldown = 0
        self.left_cooldown = 0

        # Detection threshold
        self.threshold = self.setting.inference.confidence_threshold
        self.stability_window = self.setting.inference.stability_window

        # Last results for display
        self.res_right = np.zeros(len(self.gesture_classes))
        self.res_left = np.zeros(len(self.gesture_classes))

        # Gesture history for display
        self.gesture_display_history = deque(maxlen=5)

        # FPS
        self._fps_start_time = time.time()
        self._fps_counter = 0
        self.current_fps = 0.0

        # ArUco screen detector for touch-to-cursor mapping
        self.aruco_detector = aruco_detector
        self._touch_cursor_enabled = aruco_detector is not None

        # Current finger positions (updated each frame)
        self._right_index_tip: Optional[Tuple[float, float]] = None
        self._left_index_tip: Optional[Tuple[float, float]] = None
        self._frame_size: Tuple[int, int] = (1280, 720)  # Default, updated on first frame

        if self._touch_cursor_enabled:
            self.logger.info(f"[GestureDetector] Touch-to-cursor enabled via ArUco")

        # Spatial-based hand tracker (replaces MediaPipe's unreliable handedness)
        self.hand_tracker = HandTracker()

        # OneEuroFilters for smoothing
        # Parameters tuned for responsiveness with smooth motion
        self._filter_min_cutoff = 1.4   # Lower = smoother, higher = more responsive
        self._filter_beta = 0.07      # Higher = more reactive to fast movements
        self._filter_d_cutoff = 1.0

        # Filters for index finger tip positions (stabilize jitter from MediaPipe)
        curr_time = time.time()
        self._right_tip_filter_x = OneEuroFilter(curr_time, 0.5, min_cutoff=self._filter_min_cutoff, beta=self._filter_beta, d_cutoff=self._filter_d_cutoff)
        self._right_tip_filter_y = OneEuroFilter(curr_time, 0.5, min_cutoff=self._filter_min_cutoff, beta=self._filter_beta, d_cutoff=self._filter_d_cutoff)
        self._left_tip_filter_x = OneEuroFilter(curr_time, 0.5, min_cutoff=self._filter_min_cutoff, beta=self._filter_beta, d_cutoff=self._filter_d_cutoff)
        self._left_tip_filter_y = OneEuroFilter(curr_time, 0.5, min_cutoff=self._filter_min_cutoff, beta=self._filter_beta, d_cutoff=self._filter_d_cutoff)

        # Drag state tracking for touch_hold gesture
        self._is_dragging = False
        self._last_drag_pos: Optional[Tuple[int, int]] = None

        # Macropad interaction flag - when True, skip cursor movements
        # This is set by detection_window when finger is over macropad
        self._macropad_active = False

        # Finger tip position cache for stable touch detection
        # The finger tip jitters during touch, so we use the position from
        # a few frames BEFORE the touch gesture is detected
        self.TOUCH_CACHE_SIZE = 8          # Keep last 8 frames
        self.TOUCH_CACHE_LOOKBACK = 6      # Use position from 4 frames ago
        self._right_tip_cache: deque = deque(maxlen=self.TOUCH_CACHE_SIZE)
        self._left_tip_cache: deque = deque(maxlen=self.TOUCH_CACHE_SIZE)

    def _load_model(self):
        """Load TFLite models."""
        if os.path.exists(self.model_path):
            try:
                interpreter = tf.lite.Interpreter(model_path=self.model_path)
                interpreter.allocate_tensors()

                input_details = interpreter.get_input_details()[0]
                output_details = interpreter.get_output_details()[0]

                self.interpreter = {
                    'interpreter': interpreter,
                    'input': input_details,
                    'output': output_details
                }

                self.logger.info(f"[GestureDetector] Loaded model: {self.model_path}")
                self.logger.info(f"[GestureDetector]   Input shape: {input_details['shape']}")
                self.logger.info(f"[GestureDetector]   Output shape: {output_details['shape']}")
            except Exception as e:
                self.logger.info(f"[GestureDetector] Error loading model: {e}")
        else:
            self.logger.info(f"[GestureDetector] Model not found: {self.model_path}")

    def set_aruco_detector(self, detector: "ArUcoScreenDetector") -> None:
        """Set or update the ArUco screen detector for touch-to-cursor mapping."""
        self.aruco_detector = detector
        self._touch_cursor_enabled = detector is not None
        if self._touch_cursor_enabled:
            self.logger.info(f"[GestureDetector] ArUco detector attached")

    def set_macropad_active(self, active: bool) -> None:
        """
        Set whether macropad interaction is active.
        When True, cursor movements from touch gestures are disabled
        to let the macropad handle the interaction instead.
        """
        self._macropad_active = active

    def _get_cached_tip(self, hand: str) -> Optional[Tuple[float, float]]:
        """
        Get finger tip position from cache (a few frames before current).

        When performing a touch gesture, the finger tip jitters at the moment of touch.
        Using a cached position from a few frames earlier gives more stable/accurate results.

        Args:
            hand: "Right" or "Left"

        Returns:
            Cached (x, y) position or None if cache is insufficient
        """
        cache = self._right_tip_cache if hand == "Right" else self._left_tip_cache

        if len(cache) < self.TOUCH_CACHE_LOOKBACK:
            # Not enough history, return current position as fallback
            return self._right_index_tip if hand == "Right" else self._left_index_tip

        # Get position from TOUCH_CACHE_LOOKBACK frames ago
        # cache[-1] is current, cache[-2] is 1 frame ago, etc.
        lookback_idx = -self.TOUCH_CACHE_LOOKBACK
        return cache[lookback_idx]

    def _click_at_touch(self, hand: str) -> bool:
        """
        Move cursor to RIGHT hand index finger tip and click using ArUco homography.
        Only works when ArUco screen is detected.
        Only triggered when touch gesture is logged (after cooldown).

        Uses CACHED position from a few frames before touch to avoid jitter.
        """
        # Skip if macropad is handling interaction
        if self._macropad_active:
            self.logger.debug("[Touch] Skipped - macropad is active")
            return False

        # Only use right hand
        if hand != "Right":
            return False

        # Must have ArUco detector
        if self.aruco_detector is None:
            self.logger.info("[Touch] No ArUco detector")
            return False

        # Must have valid screen detection
        if not self.aruco_detector.is_valid:
            self.logger.info("[Touch] ArUco screen not detected")
            return False

        # Get CACHED finger tip position (more stable than current)
        tip = self._get_cached_tip(hand)
        if tip is None:
            self.logger.info("[Touch] No finger tip position")
            return False

        # Convert normalized (0-1) to camera pixels
        cam_w, cam_h = self._frame_size
        finger_x = tip[0] * cam_w
        finger_y = tip[1] * cam_h

        self.logger.info(f"[Touch] Using cached tip: norm=({tip[0]:.3f}, {tip[1]:.3f}) -> cam_px=({finger_x:.1f}, {finger_y:.1f})")

        # Transform to screen coordinates via homography
        screen_pos = self.aruco_detector.transform_point((finger_x, finger_y))

        if screen_pos is None:
            self.logger.info("[Touch] Homography transform failed")
            return False

        screen_x, screen_y = screen_pos
        self.logger.info(f"[Touch] Screen pos: ({screen_x}, {screen_y}) -> Click!")

        # Move cursor and click
        ActionExecutor.click_at(screen_x, screen_y)

        return True

    def _move_on_hover(self, hand: str) -> bool:
        """
        Move cursor to RIGHT hand index finger tip using ArUco homography.
        Only works when ArUco screen is detected.
        """
        # Skip if macropad is handling interaction
        if self._macropad_active:
            self.logger.debug("[Touch_hover] Skipped - macropad is active")
            return False

        # Only use right hand
        if hand != "Right":
            return False

        # Must have ArUco detector
        if self.aruco_detector is None:
            self.logger.info("[Touch_hover] No ArUco detector")
            return False

        # Must have valid screen detection
        if not self.aruco_detector.is_valid:
            self.logger.info("[Touch_hover] ArUco screen not detected")
            return False

        # Get right hand index finger tip
        tip = self._right_index_tip
        if tip is None:
            self.logger.info("[Touch_hover] No finger tip position")
            return False

        # Convert normalized (0-1) to camera pixels
        cam_w, cam_h = self._frame_size
        finger_x = tip[0] * cam_w
        finger_y = tip[1] * cam_h

        self.logger.info(f"[Touch_hover] Finger tip: norm=({tip[0]:.3f}, {tip[1]:.3f}) -> cam_px=({finger_x:.1f}, {finger_y:.1f})")

        # Transform to screen coordinates via homography
        screen_pos = self.aruco_detector.transform_point((finger_x, finger_y))

        if screen_pos is None:
            self.logger.info("[Touch_hover] Homography transform failed")
            return False

        screen_x, screen_y = screen_pos

        final_x, final_y = int(screen_x), int(screen_y)

        self.logger.info(f"[Touch_hover] Screen pos: ({final_x}, {final_y})")

        # Move cursor directly
        ActionExecutor.move_cursor(final_x, final_y)
        
        return True

    def _drag_on_hold(self, hand: str) -> bool:
        """
        Drag with RIGHT hand index finger tip using ArUco homography.
        First call presses mouse down, subsequent calls move while dragging.
        Used for touch_hold gesture to drag files/items.
        """
        # Skip if macropad is handling interaction
        if self._macropad_active:
            self.logger.debug("[Touch_hold] Skipped - macropad is active")
            return False

        # Only use right hand
        if hand != "Right":
            return False

        # Must have ArUco detector
        if self.aruco_detector is None:
            self.logger.info("[Touch_hold] No ArUco detector")
            return False

        # Must have valid screen detection
        if not self.aruco_detector.is_valid:
            self.logger.info("[Touch_hold] ArUco screen not detected")
            return False

        # Get right hand index finger tip
        tip = self._right_index_tip
        if tip is None:
            self.logger.info("[Touch_hold] No finger tip position")
            return False

        # Convert normalized (0-1) to camera pixels
        cam_w, cam_h = self._frame_size
        finger_x = tip[0] * cam_w
        finger_y = tip[1] * cam_h

        # Transform to screen coordinates via homography
        screen_pos = self.aruco_detector.transform_point((finger_x, finger_y))

        if screen_pos is None:
            self.logger.info("[Touch_hold] Homography transform failed")
            return False

        screen_x, screen_y = screen_pos

        # Finger tip is already smoothed in process_frame
        final_x, final_y = int(screen_x), int(screen_y)

        if not self._is_dragging:
            # First frame of drag - press mouse down
            self.logger.info(f"[Touch_hold] Starting drag at ({final_x}, {final_y})")
            ActionExecutor.mouse_down(final_x, final_y)
            self._is_dragging = True
            self._last_drag_pos = (final_x, final_y)
        else:
            # Continue dragging - move with mouse held
            self.logger.info(f"[Touch_hold] Dragging to ({final_x}, {final_y})")
            ActionExecutor.drag_move(final_x, final_y)
            self._last_drag_pos = (final_x, final_y)
        
        return True

    def _end_drag(self):
        """Release mouse button if currently dragging."""
        if self._is_dragging:
            if self._last_drag_pos:
                x, y = self._last_drag_pos
                self.logger.info(f"[Touch_hold] Ending drag at ({x}, {y})")
                ActionExecutor.mouse_up(x, y)
            else:
                # Fallback - release at current cursor position
                import pyautogui
                pos = pyautogui.position()
                ActionExecutor.mouse_up(pos[0], pos[1])
            self._is_dragging = False
            self._last_drag_pos = None


    def _extract_raw_keypoints(self, results, flip_h: bool, swap_hands: bool):
        """
        Extract keypoints using spatial-based hand tracking.
        The HandTracker maintains stable handedness labels by tracking hands
        via centroid position continuity, ignoring MediaPipe's unreliable labels.
        """
        return self.hand_tracker.update(results, flip_h, swap_hands)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Process a single video frame.
        """
        # Store frame size for coordinate conversion
        self._frame_size = (frame.shape[1], frame.shape[0])  # (width, height)

        # Reset finger positions (will be updated if hands detected)
        self._right_index_tip = None
        self._left_index_tip = None

        # FPS
        self._fps_counter += 1
        elapsed = time.time() - self._fps_start_time
        if elapsed > 1.0:
            self.current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_start_time = time.time()

        # Get setting
        flip_h = self.setting.camera.flip_horizontal
        swap_hands = self.setting.camera.swap_hands

        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True

        # Convert back for drawing
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        detections = {}

        # Extract keypoints FIRST using spatial hand tracker
        # This populates the tracker with stable hand labels before we draw
        right_kp, left_kp = self._extract_raw_keypoints(results, flip_h, swap_hands)

        # Draw landmarks and labels using tracker's stable assignments
        if results.multi_hand_landmarks:
            # Get stable hand labels from tracker
            hand_labels = self.hand_tracker.get_hand_labels(swap_hands)

            for hand_landmarks in results.multi_hand_landmarks:
                # Draw skeleton
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Draw stable labels at wrist positions
            # The tracker knows which hand is which by centroid matching
            for _, (label, centroid) in hand_labels.items():
                h, w, _ = image.shape
                wx, wy = int(centroid[0] * w), int(centroid[1] * h)
                color = (255, 100, 100) if label == "Right" else (100, 100, 255)
                cv2.putText(image, label, (wx - 30, wy + 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Store finger tip positions using tracker's stable assignments
        right_tip, left_tip = self.hand_tracker.get_finger_tips(swap_hands)
        curr_time = time.time()

        if right_tip is not None:
            smoothed_x = self._right_tip_filter_x(curr_time, right_tip[0])
            smoothed_y = self._right_tip_filter_y(curr_time, right_tip[1])
            self._right_index_tip = (smoothed_x, smoothed_y)
            # Add to cache for stable touch detection
            self._right_tip_cache.append((smoothed_x, smoothed_y))

        if left_tip is not None:
            smoothed_x = self._left_tip_filter_x(curr_time, left_tip[0])
            smoothed_y = self._left_tip_filter_y(curr_time, left_tip[1])
            self._left_index_tip = (smoothed_x, smoothed_y)
            # Add to cache for stable touch detection
            self._left_tip_cache.append((smoothed_x, smoothed_y))

        # Update sequences
        if right_kp is not None and np.any(right_kp):
            self.right_sequence.append(right_kp)
            self.right_sequence = self.right_sequence[-self.sequence_length:]
            self.right_lock = True

        if left_kp is not None and np.any(left_kp):
            self.left_sequence.append(left_kp)
            self.left_sequence = self.left_sequence[-self.sequence_length:]
            self.left_lock = True

        # Right hand prediction
        if self.right_lock and len(self.right_sequence) == self.sequence_length:
            if self.interpreter:
                seq_array = np.array(self.right_sequence)  # (seq_len, 84)
                features = self.feature_engineer.transform(seq_array)  # (seq_len, 67)

                gesture, confidence, self.res_right = self._predict('Right', features)
                detections['Right'] = {'gesture': gesture, 'confidence': confidence}

                # Handle gesture
                self._handle_gesture('Right', gesture, confidence)

        # Left hand prediction
        if self.left_lock and len(self.left_sequence) == self.sequence_length:
            if self.interpreter:
                seq_array = np.array(self.left_sequence)
                features = self.feature_engineer.transform(seq_array)

                gesture, confidence, self.res_left = self._predict('Left', features)
                detections['Left'] = {'gesture': gesture, 'confidence': confidence}

                self._handle_gesture('Left', gesture, confidence)

        # Draw probability bars (before resetting locks)
        image = self._draw_prob_bars(image, self.right_lock, self.left_lock)

        # Draw gesture history
        image = self._draw_history(image)

        # Draw FPS
        img_h, img_w, _ = image.shape
        cv2.putText(image, f"FPS: {self.current_fps:.1f}", (10, img_h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Reset locks
        self.right_lock = False
        self.left_lock = False

        # Add index tips to detections dict using tracker's stable assignments
        if 'Right' in detections and right_tip is not None:
            detections['Right']['index_tip'] = (right_tip[0], right_tip[1], 0.0)
        if 'Left' in detections and left_tip is not None:
            detections['Left']['index_tip'] = (left_tip[0], left_tip[1], 0.0)

        return image, detections

    def _predict(self, hand: str, features: np.ndarray):
        """Run TFLite inference."""
        interpreter = self.interpreter['interpreter']
        input_details = self.interpreter['input']
        output_details = self.interpreter['output']

        # Add batch dimension: (seq_len, 67) -> (1, seq_len, 67)
        input_data = np.expand_dims(features, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])[0]  # Remove batch dim

        # Get prediction
        class_idx = int(np.argmax(output))
        confidence = float(output[class_idx])

        # Smoothing
        if hand == 'Right':
            self.right_predictions.append(class_idx)
            preds = list(self.right_predictions)
        else:
            self.left_predictions.append(class_idx)
            preds = list(self.left_predictions)

        # Most common in last [# of stability_window] predictions
        last_preds = preds[-self.stability_window:]
        if last_preds:
            most_common = Counter(last_preds).most_common(1)[0][0]
            if most_common == class_idx and confidence > self.threshold:
                gesture_name = self.gesture_classes[class_idx]
                return gesture_name, confidence, output

        return "none", 0.0, output

    def _handle_gesture(self, hand: str, gesture: str, confidence: float):
        """Execute action for detected gesture."""
        # Always decrement cooldown first 
        if hand == 'Right':
            if self.right_cooldown > 0:
                self.right_cooldown -= 1
        else:
            if self.left_cooldown > 0:
                self.left_cooldown -= 1

        # "none" gesture: end any drag and skip
        if gesture == "none":
            self._end_drag()
            return

        # Below threshold: end drag immediately and skip
        if confidence < self.threshold:
            self._end_drag()
            return

        if gesture == "touch_hover":
            # End any active drag when switching to hover
            self._end_drag()
            if hand == 'Right':
                self.right_cooldown = 0
            else: 
                self.left_cooldown = 0
            self._move_on_hover(hand)
            return
        elif gesture == "touch_hold":
            # Drag gesture - mouse down and move
            if hand == 'Right':
                self.right_cooldown = 0
            else: 
                self.left_cooldown = 0
            self._drag_on_hold(hand)
            return
        else:
            # Any other gesture ends the drag
            self._end_drag()

        # Check cooldown (applies to all gestures including touch)
        # Only trigger action when cooldown expires (gesture is logged)
        if hand == 'Right':
            if self.right_cooldown > 0:
                return  # Still on cooldown, skip
            self.right_cooldown = self.COOLDOWN_FRAMES
        else:
            if self.left_cooldown > 0:
                return  # Still on cooldown, skip
            self.left_cooldown = self.COOLDOWN_FRAMES

        # Log detection (all gestures logged the same way)
        timestamp = time.strftime("%H:%M:%S")
        entry = f"{timestamp} {hand}: {gesture} ({confidence:.2f})"
        self.gesture_display_history.append(entry)
        self.logger.info(f"[Gesture] {entry}")

        # Special action for touch - move cursor and click
        if gesture == "touch":
            self._click_at_touch(hand)
            return

        # Execute mapped action for other gestures
        mapping_key = f"{hand}_{gesture}"
        actions = self.setting.get_gesture_actions(mapping_key)
        if actions:
            # Filter out "none" actions
            valid_actions = [a for a in actions if a.type != "none"]
            if valid_actions:
                if len(valid_actions) == 1:
                    # Single action - execute directly
                    self.executor.execute(valid_actions[0].type, valid_actions[0].value)
                else:
                    # Multiple actions - execute as sequence with delays
                    self.executor.execute_sequence(valid_actions)

    def _draw_prob_bars(self, image: np.ndarray, right_active: bool, left_active: bool) -> np.ndarray:
        """Draw probability visualization bars"""
        h, w, _ = image.shape
        colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 200, 16),
                  (200, 117, 245), (16, 245, 200), (245, 16, 117), (117, 16, 245)]

        # Right hand bars (right side of screen)
        if right_active:
            for num, prob in enumerate(self.res_right):
                bar_length = int(prob * 100)
                margin = 10

                # Right-aligned rectangle
                x2 = w - margin
                x1 = x2 - bar_length
                y1 = 60 + num * 40
                y2 = 90 + num * 40

                color = colors[num % len(colors)]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)

                # Right-aligned text
                label = self.gesture_classes[num] if num < len(self.gesture_classes) else f"class_{num}"
                text = f"{label}"
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                text_x = w - text_w - margin
                text_y = 85 + num * 40

                cv2.putText(image, text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Left hand bars (left side of screen)
        if left_active:
            for num, prob in enumerate(self.res_left):
                bar_length = int(prob * 100)

                # Left-aligned rectangle
                x1 = 0
                x2 = bar_length
                y1 = 60 + num * 40
                y2 = 90 + num * 40

                color = colors[num % len(colors)]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)

                # Left-aligned text
                label = self.gesture_classes[num] if num < len(self.gesture_classes) else f"class_{num}"
                cv2.putText(image, label, (0, 85 + num * 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return image

    def _draw_history(self, image: np.ndarray) -> np.ndarray:
        """Draw gesture history."""
        h, w, _ = image.shape
        x = w // 2 - 150
        y = 30

        cv2.putText(image, "Gesture History:", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        for i, entry in enumerate(reversed(list(self.gesture_display_history))):
            if i >= 5:
                break
            y_pos = y + 25 + (i * 20)
            alpha = 1.0 - (i * 0.15)
            color = (int(150 * alpha), int(255 * alpha), int(150 * alpha))
            cv2.putText(image, entry, (x, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return image

    def close(self):
        """Cleanup resources."""
        self.hands.close()
