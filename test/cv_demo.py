# Copyright (c) 2026 Huynh Huy. All rights reserved.

"""
OpenCV Demo - Run this to see rectangle and text drawing in action.
"""

import cv2
import numpy as np

# Create a blank image (black background)
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# ==============================================================================
# Step 1: Draw Rectangle
# ==============================================================================
cv2.rectangle(frame, (100, 100), (300, 250), (0, 255, 0), 2)
#                    top-left     bottom-right   color BGR   thickness

# ==============================================================================
# Step 2: Draw Text
# ==============================================================================
cv2.putText(frame, 'Hello HandFlow', (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#                                    bottom-left                         scale  color BGR      thickness

# ==============================================================================
# Step 3: Show the result
# ==============================================================================
cv2.imshow('CV Demo', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
