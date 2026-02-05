# Copyright (c) 2026 Huynh Huy. All rights reserved.

"""
OpenCV Code Snippets for HandFlow Macros
========================================

Copy these code blocks into HandFlow text actions for quick insertion.
"""

# ==============================================================================
# CV Video Loop
# ==============================================================================
import cv2


cap = cv2.VideoCapture(0)
while True:
	ret, frame = cap.read()
	if not ret:
		break
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()


# ==============================================================================
# Draw Rectangle
# ==============================================================================
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                    top-left   bottom-right  color BGR  thickness


# ==============================================================================
# Draw Text
# ==============================================================================
cv2.putText(frame, 'text', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#                         bottom-left                    scale  color BGR    thickness