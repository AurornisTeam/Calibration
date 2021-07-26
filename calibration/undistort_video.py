import cv2
import numpy as np


cap = cv2.VideoCapture(1)

DIM=(1280, 720)
K=np.array([[796.2152689047866, 0.0, 666.7474094441768], [0.0, 802.2658823670218, 379.81656819013784], [0.0, 0.0, 1.0]])
D=np.array([[-0.02544263874037485], [-0.08730606939161767], [0.2367713210309387], [-0.1741615421908499]])
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
while True:
	ret, frame = cap.read()
	h,w = frame.shape[:2]
	
	undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

	cv2.imshow("undistorted_frame",undistorted_frame)
	cv2.imshow("distort",frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



