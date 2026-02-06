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
cap = cv2.VideoCapture(0)                                                                                                                          
while cap.isOpened():
	ret, frame = cap.read()                                                                                                                        
	if not ret:                                                                  
		break
	cv2.imshow('Frame', frame)
      	if cv2.waitKey(1) & 0xFF == ord('q'):
          		break
cap.release()
cv2.destroyAllWindows()


sAlign


if __name__ == "__main__":
    main()
