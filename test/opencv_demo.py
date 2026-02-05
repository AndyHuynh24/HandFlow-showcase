#!/usr/bin/env python3
# Copyright (c) 2026 Huynh Huy. All rights reserved.

"""
OpenCV Drawing Demo - MacroPad Template Demo
=============================================

Simple OpenCV demo to demonstrate typing out code templates
using MacroPad buttons & handgesture for quick coding.

"""

import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #Draw rect
        cv2.rectangle(frame, (100, 100), (650, 600), (0, 255, 0), 2)
        #                    top-left   bottom-right  color angleBGR  thickness

        #Draw text
        cv2.putText(frame, 'text', (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        #                         bottom-left                      scale  color BGR    thickness

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
