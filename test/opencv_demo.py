#!/usr/bin/env python3
# Copyright (c) 2026 Huynh Huy. All rights reserved.

"""
OpenCV Drawing Demo - MacroPad Template Demo
=============================================

Simple OpenCV demo to demonstrate typing out code templates
using MacroPad buttons for quick coding.

Run: python opencv_demo.py

MacroPad Button Suggestions:
- Button 1: cv2.rectangle(img, (x1, y1), (x2, y2), (B, G, R), thickness)
- Button 2: cv2.circle(img, (cx, cy), radius, (B, G, R), thickness)
- Button 3: cv2.line(img, (x1, y1), (x2, y2), (B, G, R), thickness)
- Button 4: cv2.putText(img, "text", (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (B, G, R), thickness)
"""

import cv2
import numpy as np

def main():
    # Create a blank canvas
    width, height = 800, 600
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)  # Dark gray background

    # ========================================
    # DEMO: Draw shapes using OpenCV
    # Use MacroPad to type these templates!
    # ========================================

    # --- RECTANGLE ---
    # Template: cv2.rectangle(img, (x1, y1), (x2, y2), (B, G, R), thickness)
    cv2.rectangle(canvas, (50, 50), (200, 150), (0, 255, 0), 2)
    cv2.putText(canvas, "Rectangle", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # --- FILLED RECTANGLE ---
    # Template: cv2.rectangle(img, (x1, y1), (x2, y2), (B, G, R), -1)
    cv2.rectangle(canvas, (250, 50), (400, 150), (255, 100, 0), -1)
    cv2.putText(canvas, "Filled Rect", (250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

    # --- CIRCLE ---
    # Template: cv2.circle(img, (cx, cy), radius, (B, G, R), thickness)
    cv2.circle(canvas, (550, 100), 50, (0, 165, 255), 2)
    cv2.putText(canvas, "Circle", (510, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    # --- FILLED CIRCLE ---
    # Template: cv2.circle(img, (cx, cy), radius, (B, G, R), -1)
    cv2.circle(canvas, (700, 100), 50, (255, 0, 255), -1)
    cv2.putText(canvas, "Filled Circle", (650, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # --- LINE ---
    # Template: cv2.line(img, (x1, y1), (x2, y2), (B, G, R), thickness)
    cv2.line(canvas, (50, 200), (200, 280), (0, 255, 255), 3)
    cv2.putText(canvas, "Line", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # --- ARROW ---
    # Template: cv2.arrowedLine(img, (x1, y1), (x2, y2), (B, G, R), thickness)
    cv2.arrowedLine(canvas, (250, 200), (400, 280), (255, 255, 0), 3)
    cv2.putText(canvas, "Arrow", (250, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # --- ELLIPSE ---
    # Template: cv2.ellipse(img, (cx, cy), (ax, ay), angle, start, end, (B, G, R), thickness)
    cv2.ellipse(canvas, (550, 240), (80, 40), 0, 0, 360, (100, 255, 100), 2)
    cv2.putText(canvas, "Ellipse", (500, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    # --- POLYGON ---
    # Template: cv2.polylines(img, [pts], isClosed, (B, G, R), thickness)
    pts = np.array([[700, 200], [750, 250], [700, 300], [650, 250]], np.int32)
    cv2.polylines(canvas, [pts], True, (255, 128, 128), 2)
    cv2.putText(canvas, "Polygon", (660, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 128), 1)

    # --- TEXT ---
    # Template: cv2.putText(img, "text", (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (B, G, R), thickness)
    cv2.putText(canvas, "OpenCV Drawing Demo", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(canvas, "Use MacroPad to type templates quickly!", (50, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 1)

    # --- BOUNDING BOX with LABEL (Common pattern) ---
    # Template for object detection visualization
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = 50, 420, 300, 550
    label = "Detected Object: 95%"
    cv2.rectangle(canvas, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 255, 0), 2)
    cv2.rectangle(canvas, (bbox_x1, bbox_y1 - 25), (bbox_x1 + 200, bbox_y1), (0, 255, 0), -1)
    cv2.putText(canvas, label, (bbox_x1 + 5, bbox_y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # --- CROSSHAIR ---
    cx, cy = 550, 480
    cv2.line(canvas, (cx - 30, cy), (cx + 30, cy), (0, 0, 255), 2)
    cv2.line(canvas, (cx, cy - 30), (cx, cy + 30), (0, 0, 255), 2)
    cv2.circle(canvas, (cx, cy), 20, (0, 0, 255), 1)
    cv2.putText(canvas, "Crosshair", (cx - 30, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # ========================================
    # QUICK TEMPLATES (for MacroPad buttons)
    # ========================================
    templates = """
    =============== QUICK TEMPLATES ===============

    cv2.rectangle(img, (x1, y1), (x2, y2), (B, G, R), thickness)
    cv2.circle(img, (cx, cy), radius, (B, G, R), thickness)
    cv2.line(img, (x1, y1), (x2, y2), (B, G, R), thickness)
    cv2.putText(img, "text", (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (B, G, R), thickness)
    cv2.ellipse(img, (cx, cy), (ax, ay), angle, 0, 360, (B, G, R), thickness)
    cv2.arrowedLine(img, (x1, y1), (x2, y2), (B, G, R), thickness)
    cv2.polylines(img, [pts], isClosed, (B, G, R), thickness)
    cv2.fillPoly(img, [pts], (B, G, R))

    # Read/Write
    img = cv2.imread("path/to/image.jpg")
    cv2.imwrite("output.jpg", img)

    # Video capture
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    # Window
    cv2.imshow("Window", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ===============================================
    """
    print(templates)

    # Show the canvas
    cv2.imshow("OpenCV Drawing Demo - Press any key to exit", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
