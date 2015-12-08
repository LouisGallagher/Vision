import numpy as np, cv2, sys, getopt

Height = 9
Width =  6
count = 0  ## number of chessboards recognised 
objpoints = []  
imgpoints = []	  

object_pts = np.zeros((Height* Width, 3), np.float32)
object_pts[:,:2] = np.mgrid[0:Height, 0:Width].T.reshape(-1,2)

## set up calibration dstructures 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  ## for sub-pixel refinement

frame_ret = False
corners_ret = False
frame = None 
corners  = None

# this function can be used to calibrate the camera 

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
			corners_ret, corners = cv2.findChessboardCorners(gray, (Height, Width), None) # find chessboard corners in frame
			cv2.drawChessboardCorners(frame, (Height,Width), corners, corners_ret)        # draw corners on framw 
			
			cv2.putText(frame, "When corners found press space to capture sample(q to quit)", (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255))
			cv2.putText(frame, "captured: {0}/10".format(count), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255))
			
			cv2.imshow('calib', frame)
			
			if corners_ret == True and cv2.waitKey(200) ==  32:  # if corners found add correspondences  
				objpoints.append(object_pts)
				#corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
				imgpoints.append(corners)
				count += 1

	cap.release()

# this function tracks the movement of a chessboard in front of the camera and renders an image over it using homography.
def homography_loop(mtx, dist, newcamermtx, x, y, w, h, file_path):
	global frame_ret 
	global corners_ret
	global frame
	global corners
	global count 


	board = cv2.imread('Data\pattern.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
	im = cv2.imread(file_path) 
 	corners_ret, corners = cv2.findChessboardCorners(board, (Height, Width), None) # points on one plane for homography 
	
	# set up video capture
	cap = cv2.VideoCapture(0) 
	
	if not(cap.isOpened()):
		cap.open()	

	while cv2.waitKey(5) != ord('q'): 
		frame_ret, frame = cap.read()
		
		if frame_ret == True:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			corners2_ret, corners2 = cv2.findChessboardCorners(gray, (Height, Width), None)
			
			if corners2_ret == True:
				undist = cv2.undistort(gray, mtx, dist, None, newcamermtx)
				#undist= undist[y:y+h, x:x+w]
				corners2_ret, corners2 = cv2.findChessboardCorners(undist, (Height, Width), None) # points on the other plane 

				if corners2_ret:

					# compute perspective transform and apply to im
					h = cv2.findHomography(corners, corners2)[0]   
					out	 = cv2.warpPerspective(im, h,(gray.shape[1], gray.shape[0]))
					
					# create mask and mask inverse 
					gray_out = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)					
					ret, mask = cv2.threshold(gray_out, 10, 255, cv2.THRESH_BINARY)
					inv_mask = cv2.bitwise_not(mask)

					# create place for the warped image in the frame
					frame = cv2.bitwise_and(frame, frame, mask=inv_mask)
					cv2.imshow('masked frame', frame)

					# grab only the ROI from the warped image
					out = cv2.bitwise_and(out, out, mask = mask)
					cv2.imshow('masked picture', out)
					
					# combine the two to create AR effect 
					frame = cv2.add(frame, out)
 					cv2.imshow('warp', frame)
			
		cv2.imshow('calib', frame)			
	cap.release()


def main(argv):
	global count 
	global imgpoints
	global objpoints

	# grab command line args 
	try:
		opts, args = getopt.getopt(argv, "hi:", ["help", "image_path="])
	except getopt.GetoptError as err:
		print err
		sys.exit(2)
	
	file_path = None

	for o, a in opts:
		if o == '-i':
			file_path = a
		elif o in ('-h', '--help'):
			print '-i <image_path> --image_path = <image_path>'
			sys.exit(2)


	cv2.namedWindow('calib' , cv2.WINDOW_AUTOSIZE)	

	# calibrate the camera 
	calib_loop()

	if count == 10:		# i.e. camera calibration achieved

		# dimension of the frames 		
		d = (np.shape(frame)[1], np.shape(frame)[0])

		# Get camera matrix etc.
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, d, None, None) # get calibration parameters 

		newcamermtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, d, 0, d )
		x,y,w,h = roi
		
		# run homography loop
		homography_loop(mtx, dist,newcamermtx, x, y, w, h, file_path) 

	cv2.destroyAllWindows()
	

if __name__ == "__main__":
	main(sys.argv[1:])