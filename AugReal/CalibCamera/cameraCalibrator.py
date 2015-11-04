import numpy as np, cv2, sys, getopt

def camera_calib():
	## set up calibration dstructures 
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	object_pts = np.zeros((6*7, 3), np.float32)

	object_pts[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)

	objpoints = []  
	imgpoints = []	  

	## set up video capture
	cap = cv2.VideoCapture(0) 
	
	if not(cap.isOpened()):
		cap.open()

	##number of chessboards detected 	
	count = 0

	while count < 10:  ##make this configurable via parameter
		frame_ret, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		if frame_ret == True:
			corners_ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

			if corners_ret == True:
				




def main(argv):
	outFileName = ''   
	outDirName  = ''
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


if __name__ == "__main__":
	main(sys.argv[1:])