import cv2
import numpy as np
import math
from scipy.spatial import distance as dist

import glob

images = glob.glob('./*.jpg')

resim = cv2.imread("12_1280x720.jpg")
resim = cv2.resize(resim,(640,480))

blurred_resim = cv2.GaussianBlur(resim, (15, 15), 0)

hsv = cv2.cvtColor(blurred_resim,cv2.COLOR_BGR2HSV)

lower_red = np.array([0,120,70])
upper_red = np.array([10,255,255])
mask1 = cv2.inRange(hsv , lower_red , upper_red)

lower_red = np.array([160, 120, 70])
upper_red = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv, lower_red, upper_red)

mask = mask1 + mask2

contours , _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

#cv2.drawContours(blurred_resim, contours, -1, (0, 255, 0), 2)
merkez_noktasi = cv2.circle(blurred_resim, (320, 480), 7, (0, 0, 255), -1)
cv2.putText(blurred_resim, "MERKEZ", (320, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    cv2.putText(blurred_resim, "Cevre: {}".format(len(approx)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    if cv2.contourArea(contour)>30 and len(approx) >= 4:
        print("kırmızı {}".format(cv2.contourArea(contour)))
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(blurred_resim, center, radius, (0, 255, 0), 2)
        cv2.circle(blurred_resim, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(blurred_resim, "Merkez " + str(cx) + "," + str(cy), (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.line(blurred_resim, (cx, cy), (320, 480), (255, 0, 0), 1)
        piksel_uzaklik = dist.euclidean((cx,cy), (320,480))
        print("mavi piksel {}".format(piksel_uzaklik))

#cv2.imshow("Resim",resim)
cv2.imshow("B resim",blurred_resim)
cv2.imshow("mask",mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
