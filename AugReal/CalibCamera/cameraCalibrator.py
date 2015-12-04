import numpy as np, cv2, sys, getopt

Height = 9
Width =  6
count = 0  ## number of chessboards recognised 
objpoints = []  
imgpoints = []	  

object_pts = np.zeros((Height* Width, 3), np.float32)
object_pts[:,:2] = np.mgrid[0:Height, 0:Width].T.reshape(-1,2)

## set up calibration dstructures 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

frame_ret = False
corners_ret = False
frame = None 
corners  = None



# def camera_calibrator(): ## move this functionality into calib_lo
# 	global count
# 	global objpoints
# 	global object_pts
# 	global imgpoints
# 	global corners_ret
# 	global corners
# 	global criteria
# 	global frame_ret

# 	if corners_ret == True and frame_ret == True:
# 		objpoints.append(object_pts)
# 		#corners = cv2.cornerSubPix(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), corners, (11,11), (-1,-1), criteria)
# 		imgpoints.append(corners)
# 		count += 1
		

def calib_loop():
	global frame_ret 
	global corners_ret
	global frame
	global corners
	global count 
	global objpoints
	global object_pts
	global imgpoints
	global criteria

	## set up video capture
	cap = cv2.VideoCapture(0) 
	
	if not(cap.isOpened()):
		cap.open()	

	while cv2.waitKey(5) != ord('q') and count < 10 : 
		frame_ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		if frame_ret == True:
			corners_ret, corners = cv2.findChessboardCorners(gray, (Height, Width), None)
			cv2.drawChessboardCorners(frame, (Height,Width), corners, corners_ret)
			cv2.putText(frame, "captured: {0}/10".format(count), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255))
			cv2.imshow('calib', frame)
			if corners_ret == True and cv2.waitKey(200) ==  32:
				objpoints.append(object_pts)
				#corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
				imgpoints.append(corners)
				count += 1

	cap.release()

def homography_loop(mtx, dist, newcamermtx, x, y, w, h):
	global frame_ret 
	global corners_ret
	global frame
	global corners
	global count 


	board = cv2.imread('Data\pattern.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
	im = cv2.imread('Data\color.jpg')
	corners_ret, corners = cv2.findChessboardCorners(board, (Height, Width), None)
	
	# set up video capture
	cap = cv2.VideoCapture(0) 
	
	if not(cap.isOpened()):
		cap.open()	

	while cv2.waitKey(5) != ord('q'): 
		frame_ret, frame = cap.read()
		
		if frame_ret == True:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			corners2_ret, corners2 = cv2.findChessboardCorners(gray, (Height, Width), None)
			cv2.imshow('gray', gray)

			if corners2_ret == True:
				undist = cv2.undistort(gray, mtx, dist, None, newcamermtx)
				#undist= undist[y:y+h, x:x+w]
				corners2_ret, corners2 = cv2.findChessboardCorners(undist, (Height, Width), None)

				if corners2_ret:
					h = cv2.findHomography(corners, corners2)[0]
					out = cv2.warpPerspective(im, h,(gray.shape[1], gray.shape[0]))
					
					gray_out = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
					#neg_out = cv2.bitwise_not(gray_out)
					ret, mask = cv2.threshold(gray_out, 10, 255, cv2.THRESH_BINARY)
					inv_mask = cv2.bitwise_not(mask)

					frame = cv2.bitwise_and(frame, frame, mask=inv_mask)
					cv2.imshow('masked frame', frame)

					out = cv2.bitwise_and(out, out, mask = mask)
					#frame = cv2.bitwise_and(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), neg_out)
					cv2.imshow('masked picture', out)
					frame = cv2.add(frame, out)
 					cv2.imshow('warp', frame)
			
		cv2.imshow('calib', frame)			
	cap.release()


def main(argv):
	global count 
	global imgpoints
	global objpoints

	cv2.namedWindow('calib' , cv2.WINDOW_AUTOSIZE)	
	calib_loop()

	if count == 10:				
		d = (np.shape(frame)[1], np.shape(frame)[0])
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, d, None, None)

		newcamermtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, d, 0, d )
		x,y,w,h = roi
		
		homography_loop(mtx, dist,newcamermtx, x, y, w, h)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
	

if __name__ == "__main__":
	main(sys.argv[1:])