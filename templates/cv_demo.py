# Copyright (c) 2026 Huynh Huy. All rights reserved.

import cv2

cap = cv2.VideoCapture(0)
while True:
	ret, frame = cap.read()
	if not ret:
		break

	# Draw Rectangle (top-left to bottom-right, green, thickness 2)
	cv2.rectangle(frame, (100, 100), (300, 250), (0, 255, 0), 2)

	# Draw Text (bottom-left origin, white)
	cv2.putText(frame, 'Hello HandFlow', (120, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

	cv2.imshow('CV Demo', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
