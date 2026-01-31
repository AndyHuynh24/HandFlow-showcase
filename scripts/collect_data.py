"""
HandFlow GUI Data Collector
====================

Self collecitng data tool - left hand would be flip to match right hand orientation.
"""

import sys
import os
import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from handflow.utils import load_config, load_setting

class GUIDataCollector:
    def __init__(self, root):
        self.root = root
        self.root.title("HandFlow Data Collector")
        self.root.geometry("400x500")
        
        self.config = load_config("config/config.yaml")
        self.setting = load_setting("config/handflow_setting.yaml")
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # UI Variables
        self.hand_var = tk.StringVar(value="right")
        self.gesture_var = tk.StringVar()
        self.camera_var = tk.StringVar(value=str(self.setting.camera.index))
        self.batch_size_var = tk.IntVar(value=30)
        self.flip_h_var = tk.BooleanVar(value=self.setting.camera.flip_horizontal)
        self.flip_v_var = tk.BooleanVar(value=self.setting.camera.flip_vertical)
        self.swap_hands_var = tk.BooleanVar(value=False)
        
        self.trigger_record = False # Flag to bridge UI events to loop
        
        self.setup_ui()
        # self.setup_bindings() 
        
    def setup_ui(self):
        # -------------------------------------------------
        # 1. Settings Frame
        # -------------------------------------------------
        settings_frame = ttk.LabelFrame(self.root, text="Settings", padding=10)
        settings_frame.pack(fill="x", padx=10, pady=5)
        
        # Hand Selection
        ttk.Label(settings_frame, text="Hand:").grid(row=0, column=0, sticky="w", pady=5)
        frame_hand = ttk.Frame(settings_frame)
        frame_hand.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(frame_hand, text="Right", variable=self.hand_var, value="right", command=self.update_gestures).pack(side="left", padx=5)
        ttk.Radiobutton(frame_hand, text="Left", variable=self.hand_var, value="left", command=self.update_gestures).pack(side="left", padx=5)
        
        # Gesture Selection
        ttk.Label(settings_frame, text="Gesture:").grid(row=1, column=0, sticky="w", pady=5)
        self.combo_gesture = ttk.Combobox(settings_frame, textvariable=self.gesture_var, state="readonly")
        self.combo_gesture.grid(row=1, column=1, sticky="ew", padx=5)
        
        # Camera Selection
        ttk.Label(settings_frame, text="Camera:").grid(row=2, column=0, sticky="w", pady=5)
        self.combo_camera = ttk.Combobox(settings_frame, textvariable=self.camera_var, 
                                       values=[str(i) for i in range(5)], state="readonly", width=5)
        self.combo_camera.grid(row=2, column=1, sticky="w", padx=5)

        # Flip Options
        ttk.Label(settings_frame, text="Flip:").grid(row=3, column=0, sticky="w", pady=5)
        frame_flip = ttk.Frame(settings_frame)
        frame_flip.grid(row=3, column=1, sticky="w")
        ttk.Checkbutton(frame_flip, text="Horiz", variable=self.flip_h_var).pack(side="left", padx=2)
        ttk.Checkbutton(frame_flip, text="Vert", variable=self.flip_v_var).pack(side="left", padx=2)

        # Swap Handedness
        ttk.Label(settings_frame, text="Swap:").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Checkbutton(settings_frame, text="Swap L/R Labels", variable=self.swap_hands_var).grid(row=4, column=1, sticky="w", padx=5)

        # Batch Size
        ttk.Label(settings_frame, text="Batch:").grid(row=5, column=0, sticky="w", pady=5)
        ttk.Spinbox(settings_frame, from_=1, to=200, textvariable=self.batch_size_var, width=5).grid(row=5, column=1, sticky="w", padx=5)

        # -------------------------------------------------
        # 2. Control Frame
        # -------------------------------------------------
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        control_frame.pack(fill="x", padx=10, pady=20)
        
        self.btn_launch = ttk.Button(control_frame, text="LAUNCH COLLECTOR", command=self.start_collection)
        self.btn_launch.pack(fill="x", pady=5)
        
        # Add a runtime Record button
        self.btn_record = ttk.Button(control_frame, text="START RECORDING (Enter/Space)", command=lambda: self.set_trigger(True))
        self.btn_record.pack(fill="x", pady=5)
        self.btn_record.state(['disabled'])

        # Initialize
        self.update_gestures()

    def update_gestures(self):
        hand = self.hand_var.get()
       
        gestures = self.config.model.gestures
     
        self.combo_gesture['values'] = gestures
        if gestures:
            self.combo_gesture.current(0)
            
    def set_trigger(self, val):
        self.trigger_record = val

    def get_next_sequence_id(self, action_path):
        if not os.path.exists(action_path):
            return 0
        dirs = [int(d) for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d)) and d.isdigit()]
        if not dirs:
            return 0
        return max(dirs) + 1

    def extract_keypoints(self, results, apply_x_flip_canon=False):
        """Extract keypoints from the detected hand (max 1 hand)."""
        if not results.multi_hand_landmarks:
            return np.zeros(21 * 4)

        # Get the only hand (max_num_hands=1)
        hand_landmarks = results.multi_hand_landmarks[0]
        kp = np.array([[res.x, res.y, res.z, 0.0] for res in hand_landmarks.landmark])
        
        # Apply x-flip canonicalization for left hand to match right hand orientation
        if apply_x_flip_canon:
            kp[:, 0] = 1.0 - kp[:, 0]
        
        return kp.flatten()

    def setup_bindings(self):
        # Global key bindings (work when GUI has focus)
        self.root.bind('<Return>', self.on_trigger)
        self.root.bind('<space>', self.on_trigger)
        self.root.bind('f', self.toggle_flip)
        self.root.bind('s', self.toggle_swap)
        self.root.bind('q', self.stop_collection)
        
    def on_trigger(self, event=None):
        if self.is_running: # Only trigger if camera is open
            if not self.is_recording:
                self.is_recording = True

    def toggle_flip(self, event=None):
        if self.is_running:
            self.flip_h_var.set(not self.flip_h_var.get())

    def toggle_swap(self, event=None):
        if self.is_running:
            self.swap_hands_var.set(not self.swap_hands_var.get())

    def start_collection(self):
        """Initialize camera and start the Tkinter-driven loop"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            return # Already running

        gesture = self.gesture_var.get()
        if not gesture:
            messagebox.showwarning("Warning", "Please select a gesture")
            return
            
        try:
            cam_idx = int(self.camera_var.get())
        except ValueError:
            cam_idx = 0
            
        self.target_batch = self.batch_size_var.get()
        self.hand_selected = self.hand_var.get()
        
        self.DATA_PATH = Path("data/raw")
        self.base_path = self.DATA_PATH / f"{self.hand_selected}_mp_data" / gesture
        self.current_sequence_id = self.get_next_sequence_id(self.base_path)
        
        self.cap = cv2.VideoCapture(cam_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open camera {cam_idx}")
            return

        self.btn_record.state(['!disabled'])
        self.btn_launch.state(['disabled']) # Disable launch button while running
        
        self.sequence_length = self.config.data.sequence_length
        self.batch_count = 0
        self.frame_num = 0
        self.is_recording = False
        self.is_running = True
        self.zero_frame_count = 0
        
        # Initialize MediaPipe Hands 
        self.hands_module = self.mp_hands.Hands(
            min_detection_confidence=self.config.mediapipe.min_detection_confidence, 
            min_tracking_confidence=self.config.mediapipe.min_tracking_confidence,
            max_num_hands=1,
            model_complexity=self.config.mediapipe.model_complexity
        )
        
        # Start the loop
        self.process_frame()

    def stop_collection(self, event=None):
        """Cleanup and close camera"""
        if hasattr(self, 'is_running') and self.is_running:
            self.is_running = False
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            if hasattr(self, 'hands_module'):
                self.hands_module.close()
            cv2.destroyAllWindows()
            
            self.btn_record.state(['disabled'])
            self.btn_launch.state(['!disabled']) # Re-enable launch
            print("Collection Stopped.")

    def process_frame(self):
        """Single frame processing scheduled by root.after"""
        if not self.is_running or not self.cap.isOpened():
            self.stop_collection()
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.stop_collection()
            return

        # Get current settings
        flip_h = self.flip_h_var.get()
        flip_v = self.flip_v_var.get()
        swap_hands = self.swap_hands_var.get()

        # Image processing
        if flip_h:
            frame = cv2.flip(frame, 1)
        if flip_v:
            frame = cv2.flip(frame, 0)
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.hands_module.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw and Extract
        self.draw_debug(image, results, flip_h, swap_hands)
        
        # Collection Logic
        self.handle_collection(image, results, flip_h, swap_hands)
        
        cv2.imshow('HandFlow Collector (Press q in GUI to quit)', image)
        
        # Process OpenCV UI events and Input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.stop_collection()
            return
        elif key == 13 or key == 10: # Enter
            self.on_trigger()
        elif key == ord('f'):
            self.toggle_flip()
        elif key == ord('s'):
            self.toggle_swap()
        
        # Schedule next frame (minimal delay for max fps)
        self.root.after(1, self.process_frame)

    def draw_debug(self, image, results, flip_h, swap_hands):
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                wrist = hand_landmarks.landmark[0]
                h, w, c = image.shape
                cx, cy = int(wrist.x * w), int(wrist.y * h)
                if results.multi_handedness:
                    label = results.multi_handedness[idx].classification[0].label
                    if flip_h: label = "Right" if label == "Left" else "Left"
                    if swap_hands: label = "Right" if label == "Left" else "Left"
                    cv2.putText(image, f"{label} Hand", (cx, cy - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    def handle_collection(self, image, results, flip_h, swap_hands):
        gesture = self.gesture_var.get()
        
        # Check if this is a "noise" gesture that should auto-collect
        is_noise_gesture = gesture.lower() in ['none', 'nonezoom', 'touch_hover', 'touch_hold']
        
        # Check if batch is done
        if self.batch_count >= self.target_batch:
            cv2.putText(image, "BATCH COMPLETE! (Enter/Space to restart)", (50, 300), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            # Stop recording if it was somehow active
            self.is_recording = False
            return

        if not self.is_recording:
            # Auto-trigger for noise gestures
            if is_noise_gesture:
                self.is_recording = True  # Auto-start recording
                
            # Standby UI
            cv2.putText(image, f"COLLECTING: {gesture}" + (" [AUTO]" if is_noise_gesture else ""), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Batch: {self.batch_count}/{self.target_batch}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Next ID: {self.current_sequence_id}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            # Recording UI
            cv2.circle(image, (30, 30), 15, (0, 0, 255), -1)
            cv2.putText(image, f"RECORDING {self.frame_num}/{self.sequence_length}", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Save Logic
            save_path = self.base_path / str(self.current_sequence_id)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Apply x-flip canonicalization for left hand to match right hand orientation
            is_left_hand = self.hand_selected == "left"
            kp = self.extract_keypoints(results, apply_x_flip_canon=is_left_hand)
            
            if np.all(kp == 0):
                self.zero_frame_count += 1
                
            np.save(save_path / f"{self.frame_num}.npy", kp)
            self.frame_num += 1
            
            if self.frame_num >= self.sequence_length:
                if self.zero_frame_count > 12:
                    print(f"❌ Discarding {self.current_sequence_id}")
                    import shutil
                    if save_path.exists(): shutil.rmtree(save_path)
                else:
                    self.current_sequence_id += 1
                    self.batch_count += 1
                    print(f"Saved {self.current_sequence_id-1}")
                
                # Reset
                self.frame_num = 0
                self.zero_frame_count = 0
                self.is_recording = False

def main():
    root = tk.Tk()
    app = GUIDataCollector(root)
    root.mainloop()

if __name__ == "__main__":
    main()
