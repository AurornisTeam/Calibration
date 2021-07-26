import numpy as np
import cv2

# You should replace these 3 lines with the output in calibration step
DIM=(1280, 720)
K=np.array([[796.2152689047866, 0.0, 666.7474094441768], [0.0, 802.2658823670218, 379.81656819013784], [0.0, 0.0, 1.0]])
D=np.array([[-0.02544263874037485], [-0.08730606939161767], [0.2367713210309387], [-0.1741615421908499]])

img = cv2.imread('opencv_frame_30.png')
h,w = img.shape[:2]
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
cv2.imshow("undistorted", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

