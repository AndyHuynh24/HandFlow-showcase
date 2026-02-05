#!/usr/bin/env python3
"""
OpenCV Demo - Video Loop with Drawing
======================================

Demo workflow:
1. Set breakpoints on lines marked with # <-- BREAKPOINT
2. Run debug (F5)
3. Use Step Over (F10), Step Into (F11) to walk through
4. Commit to git

Press 'q' to quit the demo.
"""

import cv2
import time


def main():
    # ========================================
    # FIELD 1: VIDEO CAPTURE LOOP
    # ========================================
    cap = cv2.VideoCapture(0)  # <-- BREAKPOINT: Camera init
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()  # <-- BREAKPOINT: Frame capture
        if not ret:
            break

        # ========================================
        # FIELD 2: DRAW RECTANGLE
        # ========================================
        x, y, w, h = 100, 100, 200, 150  # <-- BREAKPOINT: Inspect values
        color = (0, 255, 0)  # Green in BGR
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # ========================================
        # FIELD 3: DRAW TEXT
        # ========================================
        text = "OpenCV Demo"  # <-- BREAKPOINT: Inspect text
        position = (10, 30)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, position, font, 1.0, (255, 255, 255), 2)

        # Show frame
        cv2.imshow('Demo', frame)  # <-- BREAKPOINT: Before display

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()  # <-- BREAKPOINT: Start here
