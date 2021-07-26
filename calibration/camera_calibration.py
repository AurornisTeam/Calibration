import numpy as np
import cv2
import glob
import yaml

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./*.png')


for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
	
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners2,ret)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
	
print(mtx) #camera matrix
print(dist) # distortion coefficient
#cv2.imshow('img',img)


#UNDISTORTION

data = {'camera_matrix': np.asarray(mtx,np.float32),
        'dist_coeff': np.asarray(dist,np.float32)}

with open("calibration_matrix.yaml", "w") as f:
    yaml.dump(data, f)


with open("calibration_matrix.yaml", "r") as f:
    data = yaml.load(f)

mtx = data['camera_matrix']
dist = data['dist_coeff']

Nimg = cv2.imread('opencv_frame_30.png')
h,  w = Nimg.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
# undistort
dst = cv2.undistort(Nimg, mtx, dist, None, newcameramtx)
# crop the image

cv2.imwrite('calibresult.png',dst)	
#print(newcameramtx)
cv2.imshow("undistorted",dst)

cv2.waitKey(0)
    
cv2.destroyAllWindows()



