import cv2, numpy as np
import glob 

#cv2.TERM_CRITERIA_EPS = epsilon ie error value has reduced to certain point
#cv2.TERM_CRITERIA_MAX_ITER = the max number of iterations has been reached
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)#when to terminate the detect function

object_pts = np.zeros((6*7, 3), np.float32)##2d array 42 * 3 array

## 7*6 Grid will have 5*6 interior squares
## we know assume the 2d plane of the board is fixed
## the below code computes the points making up the 2d plane
## that is the cornors of the interior squares
##also assume z axis stays at 0 

object_pts[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)

objpoints = []     ## 3d points in real world
imgpoints = []	   ## 2d points in image plane


images = glob.glob('Data/*.jpg')

for fname in images:
	##load up image and convert to grayscale
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	##detect Corners
	ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

	if ret == True:
		objpoints.append(object_pts)

		## refine the corner coordinates before adding them to 
		## gray = img to process
		## corners = array of initial corners
		## (11,11) = search window 
		## (-1,-1) = the zerozone in the in the middle of the search window where the computation doesn't need to be performed in this case(-1,-1) ther isn't one
		## criteria = the conditions under which to halt iterative refinement of corner position
		corners2 = cv2.cornerSubPix(gray, corners,(11,11), (-1,-1), criteria)
		imgpoints.append(corners)

		## display images 
		## cv2.drawChessboardCorners(img, (7,6), corners, ret)
		## cv2.imshow('img',img)
		## cv2.waitKey(5000)


## insert explanation of calibcamera here 		
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (640L, 480L), None, None)

newcamermtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (640L, 480L), 1, (640L, 480L))

img = cv2.imread('Data/left01.jpg')

dst = cv2.undistort(img, mtx, dist, None, newcamermtx)

x, y, w, h = roi

dst = dst[y:y+h, x:x+h]

cv2.imshow('img', dst)
cv2.waitKey(0)



cv2.destroyAllWindows()



