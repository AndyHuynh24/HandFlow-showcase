#!/usr/bin/env python3
# Copyright (c) 2026 Huynh Huy. All rights reserved.

"""
OpenCV Debugger Demo - Video Loop with Drawing
===============================================

Simple OpenCV script demonstrating video capture, drawing rectangles and text.
Use MacroPad to demonstrate debugger step/next actions.

How to use:
1. Set breakpoints on lines marked with # BREAKPOINT
2. Run in debug mode (F5 in VS Code)
3. Use MacroPad buttons for:
   - Step Into (F11): Go into function calls
   - Step Over (F10): Execute line, skip function internals
   - Step Out (Shift+F11): Exit current function
   - Continue (F5): Run to next breakpoint

MacroPad Button Suggestions:
- Button 1: F10 (Step Over)
- Button 2: F11 (Step Into)
- Button 3: Shift+F11 (Step Out)
- Button 4: F5 (Continue)
"""

import cv2
import numpy as np
import time


# ============================================
# HELPER FUNCTIONS - Step Into these
# ============================================

def draw_rectangle(frame, x, y, w, h, color, thickness=2):
    """Draw a rectangle on the frame."""
    pt1 = (x, y)  # BREAKPOINT - Inspect coordinates
    pt2 = (x + w, y + h)
    cv2.rectangle(frame, pt1, pt2, color, thickness)
    return frame


def draw_text(frame, text, x, y, color, scale=1.0, thickness=2):
    """Draw text on the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX  # BREAKPOINT - Inspect font
    position = (x, y)
    cv2.putText(frame, text, position, font, scale, color, thickness)
    return frame


def draw_circle(frame, center_x, center_y, radius, color, thickness=-1):
    """Draw a circle on the frame."""
    center = (center_x, center_y)  # BREAKPOINT - Inspect center point
    cv2.circle(frame, center, radius, color, thickness)
    return frame


def calculate_fps(prev_time):
    """Calculate frames per second."""
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)  # BREAKPOINT - Inspect FPS calculation
    return fps, current_time


# ============================================
# COLOR DEFINITIONS
# ============================================

# BGR format (Blue, Green, Red)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_MAGENTA = (255, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


# ============================================
# DEMO 1: Basic Drawing on Static Image
# ============================================

def demo_static_drawing():
    """Demo drawing shapes on a static image."""
    print("\n--- Demo 1: Static Drawing ---")

    # Create a blank image (black background)
    width, height = 640, 480
    frame = np.zeros((height, width, 3), dtype=np.uint8)  # BREAKPOINT - Inspect frame shape

    # Draw rectangles
    draw_rectangle(frame, 50, 50, 200, 100, COLOR_RED, 3)  # BREAKPOINT - Step Into
    draw_rectangle(frame, 300, 50, 150, 150, COLOR_GREEN, 2)
    draw_rectangle(frame, 100, 200, 250, 80, COLOR_BLUE, -1)  # Filled rectangle

    # Draw text
    draw_text(frame, "OpenCV Demo", 50, 350, COLOR_WHITE, 1.5, 2)  # BREAKPOINT
    draw_text(frame, "Press any key to continue", 50, 400, COLOR_YELLOW, 0.7, 1)

    # Draw circles
    draw_circle(frame, 500, 300, 50, COLOR_CYAN, -1)  # Filled
    draw_circle(frame, 500, 300, 60, COLOR_MAGENTA, 2)  # Outline

    # Display the result
    cv2.imshow("Static Drawing Demo", frame)
    cv2.waitKey(2000)  # Wait 2 seconds
    cv2.destroyAllWindows()

    print("Static drawing demo completed!")
    return frame


# ============================================
# DEMO 2: Video Capture Loop
# ============================================

def demo_video_loop():
    """Demo basic video capture loop."""
    print("\n--- Demo 2: Video Capture Loop ---")
    print("Press 'q' to quit")

    # Open camera
    cap = cv2.VideoCapture(0)  # BREAKPOINT - Camera initialization

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0
    prev_time = time.time()

    while True:
        # Read frame from camera
        ret, frame = cap.read()  # BREAKPOINT - Frame capture

        if not ret:
            print("Error: Could not read frame")
            break

        frame_count += 1

        # Calculate FPS
        fps, prev_time = calculate_fps(prev_time)  # Step Into to see FPS calculation

        # Draw FPS counter
        fps_text = f"FPS: {fps:.1f}"
        draw_text(frame, fps_text, 10, 30, COLOR_GREEN, 0.8, 2)  # BREAKPOINT

        # Draw frame counter
        frame_text = f"Frame: {frame_count}"
        draw_text(frame, frame_text, 10, 60, COLOR_YELLOW, 0.6, 1)

        # Display frame
        cv2.imshow("Video Loop Demo", frame)

        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print(f"Video loop completed! Total frames: {frame_count}")


# ============================================
# DEMO 3: Drawing Overlay on Video
# ============================================

def demo_video_with_overlay():
    """Demo drawing rectangles and text overlay on live video."""
    print("\n--- Demo 3: Video with Drawing Overlay ---")
    print("Press 'q' to quit, 'r' to toggle rectangle, 't' to toggle text")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    show_rectangle = True
    show_text = True
    prev_time = time.time()

    # Rectangle properties (can be modified during debugging)
    rect_x, rect_y = 100, 100  # BREAKPOINT - Modify these values
    rect_w, rect_h = 200, 150
    rect_color = COLOR_GREEN

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally (mirror effect)
        frame = cv2.flip(frame, 1)  # BREAKPOINT - Inspect flip operation

        # Calculate FPS
        fps, prev_time = calculate_fps(prev_time)

        # Draw rectangle overlay
        if show_rectangle:
            draw_rectangle(frame, rect_x, rect_y, rect_w, rect_h, rect_color, 3)
            # Draw corner markers
            marker_size = 10
            cv2.circle(frame, (rect_x, rect_y), marker_size, COLOR_RED, -1)
            cv2.circle(frame, (rect_x + rect_w, rect_y), marker_size, COLOR_RED, -1)
            cv2.circle(frame, (rect_x, rect_y + rect_h), marker_size, COLOR_RED, -1)
            cv2.circle(frame, (rect_x + rect_w, rect_y + rect_h), marker_size, COLOR_RED, -1)

        # Draw text overlay
        if show_text:
            draw_text(frame, "HandFlow OpenCV Demo", 10, 30, COLOR_WHITE, 0.8, 2)
            draw_text(frame, f"FPS: {fps:.1f}", 10, 60, COLOR_GREEN, 0.6, 1)
            draw_text(frame, f"Rect: {'ON' if show_rectangle else 'OFF'}", 10, 90, COLOR_YELLOW, 0.5, 1)
            draw_text(frame, "Press Q=quit, R=rect, T=text", 10, 460, COLOR_CYAN, 0.5, 1)

        cv2.imshow("Video with Overlay", frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF  # BREAKPOINT - Inspect key value
        if key == ord('q'):
            break
        elif key == ord('r'):
            show_rectangle = not show_rectangle
            print(f"Rectangle: {'ON' if show_rectangle else 'OFF'}")
        elif key == ord('t'):
            show_text = not show_text
            print(f"Text: {'ON' if show_text else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("Video overlay demo completed!")


# ============================================
# DEMO 4: Region of Interest (ROI)
# ============================================

def demo_roi_processing():
    """Demo extracting and processing region of interest."""
    print("\n--- Demo 4: Region of Interest (ROI) ---")
    print("Press 'q' to quit")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ROI coordinates
    roi_x, roi_y = 200, 150  # BREAKPOINT - ROI top-left corner
    roi_w, roi_h = 240, 180  # BREAKPOINT - ROI dimensions

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Extract ROI
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]  # BREAKPOINT - Inspect ROI

        # Process ROI (convert to grayscale, then back to BGR)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_processed = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

        # Apply edge detection to ROI
        edges = cv2.Canny(roi_gray, 50, 150)  # BREAKPOINT - Inspect edge detection
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Draw ROI rectangle on main frame
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), COLOR_GREEN, 2)
        draw_text(frame, "ROI", roi_x, roi_y - 10, COLOR_GREEN, 0.6, 1)

        # Create display with multiple views
        # Put processed ROI back
        frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = edges_colored

        draw_text(frame, "ROI Processing Demo", 10, 30, COLOR_WHITE, 0.7, 2)
        draw_text(frame, "Green box = ROI with edge detection", 10, 60, COLOR_YELLOW, 0.5, 1)

        cv2.imshow("ROI Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("ROI demo completed!")


# ============================================
# DEMO 5: Mouse Callback for Interactive Drawing
# ============================================

# Global variables for mouse demo
mouse_x, mouse_y = 0, 0
drawing = False
rectangles = []
start_point = None


def mouse_callback(event, x, y, flags, param):
    """Mouse callback function for interactive drawing."""
    global mouse_x, mouse_y, drawing, start_point, rectangles

    mouse_x, mouse_y = x, y  # BREAKPOINT - Track mouse position

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        print(f"Started drawing at ({x}, {y})")

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if start_point:
            end_point = (x, y)
            rectangles.append((start_point, end_point))  # BREAKPOINT - Inspect rectangles list
            print(f"Rectangle added: {start_point} to {end_point}")
            start_point = None


def demo_interactive_drawing():
    """Demo interactive drawing with mouse."""
    global mouse_x, mouse_y, drawing, rectangles, start_point

    print("\n--- Demo 5: Interactive Drawing ---")
    print("Click and drag to draw rectangles")
    print("Press 'c' to clear, 'q' to quit")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Reset globals
    rectangles = []
    drawing = False
    start_point = None

    cv2.namedWindow("Interactive Drawing")
    cv2.setMouseCallback("Interactive Drawing", mouse_callback)  # BREAKPOINT

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Draw all saved rectangles
        for rect in rectangles:  # BREAKPOINT - Inspect each rectangle
            pt1, pt2 = rect
            cv2.rectangle(frame, pt1, pt2, COLOR_GREEN, 2)

        # Draw current rectangle being drawn
        if drawing and start_point:
            cv2.rectangle(frame, start_point, (mouse_x, mouse_y), COLOR_YELLOW, 2)

        # Draw crosshair at mouse position
        cv2.line(frame, (mouse_x - 10, mouse_y), (mouse_x + 10, mouse_y), COLOR_RED, 1)
        cv2.line(frame, (mouse_x, mouse_y - 10), (mouse_x, mouse_y + 10), COLOR_RED, 1)

        # Draw info text
        draw_text(frame, "Interactive Drawing Demo", 10, 30, COLOR_WHITE, 0.7, 2)
        draw_text(frame, f"Mouse: ({mouse_x}, {mouse_y})", 10, 60, COLOR_CYAN, 0.5, 1)
        draw_text(frame, f"Rectangles: {len(rectangles)}", 10, 85, COLOR_YELLOW, 0.5, 1)
        draw_text(frame, "Drag to draw, C=clear, Q=quit", 10, 460, COLOR_GREEN, 0.5, 1)

        cv2.imshow("Interactive Drawing", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            rectangles = []
            print("Cleared all rectangles")

    cap.release()
    cv2.destroyAllWindows()
    print("Interactive drawing demo completed!")


# ============================================
# MAIN - Run All Demos
# ============================================

def main():
    """Main entry point - set breakpoint here to start."""
    print("=" * 50)
    print("OPENCV DEBUGGER DEMO")
    print("Video Loop, Drawing Rectangles & Text")
    print("=" * 50)
    print("\nTips:")
    print("- F10: Step Over (skip function internals)")
    print("- F11: Step Into (go inside functions)")
    print("- Shift+F11: Step Out (exit current function)")
    print("- F5: Continue (run to next breakpoint)")
    print("=" * 50)

    # BREAKPOINT HERE - Start debugging from main()

    print("\nSelect a demo to run:")
    print("1. Static Drawing (no camera)")
    print("2. Video Capture Loop")
    print("3. Video with Drawing Overlay")
    print("4. Region of Interest (ROI)")
    print("5. Interactive Drawing with Mouse")
    print("6. Run All Demos")
    print("Q. Quit")

    while True:
        choice = input("\nEnter choice (1-6 or Q): ").strip().lower()

        if choice == '1':
            demo_static_drawing()
        elif choice == '2':
            demo_video_loop()
        elif choice == '3':
            demo_video_with_overlay()
        elif choice == '4':
            demo_roi_processing()
        elif choice == '5':
            demo_interactive_drawing()
        elif choice == '6':
            demo_static_drawing()
            demo_video_loop()
            demo_video_with_overlay()
            demo_roi_processing()
            demo_interactive_drawing()
            break
        elif choice == 'q':
            break
        else:
            print("Invalid choice. Please enter 1-6 or Q.")

    print("\n" + "=" * 50)
    print("OpenCV Demo completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()  # BREAKPOINT - Start here!
