import numpy as np, cv2, sys, getopt

count = 0  ## number of chessboards recognised 
objpoints = []  
imgpoints = []	  

object_pts = np.zeros((6*5, 3), np.float32)
object_pts[:,:2] = np.mgrid[0:6, 0:5].T.reshape(-1,2)

## set up calibration dstructures 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

frame_ret = False
corners_ret = False
frame = None 
corners  = None


def camera_calibrator():
	global count
	global objpoints
	global object_pts
	global imgpoints
	global corners_ret
	global corners
	global criteria
	global frame_ret

	if corners_ret == True and frame_ret == True:
		objpoints.append(object_pts)
		#corners = cv2.cornerSubPix(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), corners, (11,11), (-1,-1), criteria)
		imgpoints.append(corners)
		count += 1
		

def calib_loop():
	global frame_ret 
	global corners_ret
	global frame
	global corners
	global count 

	## set up video capture
	cap = cv2.VideoCapture(0) 
	
	if not(cap.isOpened()):
		cap.open()	

	while cv2.waitKey(5) != ord('q') and count < 10 : 
		frame_ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		if frame_ret == True:
			corners_ret, corners = cv2.findChessboardCorners(gray, (6, 5), None)
			cv2.drawChessboardCorners(frame, (6, 5), corners, corners_ret)
			cv2.putText(frame, "captured: {0}/10".format(count), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255))
			cv2.imshow('calib', frame)
			if corners_ret == True and cv2.waitKey(200) ==  32:
				camera_calibrator()

	cap.release()

def computeHomography(mtx, dist, newcamermtx, x, y, w, h):
	global frame_ret 
	global corners_ret
	global frame
	global corners
	global count 

	## set up video capture
	cap = cv2.VideoCapture(0) 
	
	if not(cap.isOpened()):
		cap.open()	

	while cv2.waitKey(5) != ord('q'): 
		frame_ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		if frame_ret == True:
			corners_ret, corners = cv2.findChessboardCorners(gray, (6, 5), None)
			if corners_ret == True:
				gray = cv2.undistort(gray, mtx, dist, None, newcamermtx)
				gray = gray[y:y+h, x:x+w]
				# r, homog =cv2.findHomography(corners, dst)
				# if r:
				# 	cap.release() 
				# 	return homog
				
			cv2.imshow("undistorted image", dst)

	cap.release()

def main(argv):
	global count 
	global imgpoints
	global objpoints

	outFileName = ''   
	outDirName  = ''
	cv2.namedWindow('calib' , cv2.WINDOW_AUTOSIZE)
	try:
		opts, args = getopt.getopt(argv,"hf:d:")
	except:
	 	print 'cameraCalibrator.py -f<outputfilename> -d<outputdirname>'
	 	sys.exit()
	for opt, arg in opts:
	 		if opt == '-f':
	 			outFileName = arg
	 		elif opt == '-d':
	 			outDirName = arg
	 		else:
	 			print 'cameraCalibrator.py -f<outputfilename> -d<outputdirname>'
				sys.exit()

	calib_loop()
	if count == 10:
				
		d = (np.shape(frame)[1], np.shape(frame)[0])
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, d, None, None)

		newcamermtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, d, 1, d )
		x,y,w,h = roi
		#mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcamermtx, (w,h), cv2.CV_32FC2)
		homog = computeHomography(mtx, dist, newcamermtx, x, y, w, h)

	cv2.destroyAllWindows()

if __name__ == "__main__":
	main(sys.argv[1:])