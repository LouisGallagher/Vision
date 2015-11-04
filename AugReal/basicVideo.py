import cv2, numpy as np

cap = cv2.VideoCapture(0)  #can pass in a file path to video file
fourcc = cv2.cv.CV_FOURCC(*'XVID')

wr = cv2.VideoWriter('Data/vid.avi', fourcc,20.0, (640,480))					

if not(cap.isOpened()):
	cap.open()



while True:
	ret, frame = cap.read()   ##grab frame 
							  ##ret is a bool val indicating if frame returned correctly
	if ret == True:
		frame = cv2.flip(frame, 0)
		wr.write(frame)
		cv2.imshow('frame', frame)   ##show the image

		if cv2.waitKey(1) & 0xff == ord('q'):##if q key pressed exit
			break
	else:
		break


cap.release()    ##release the device 
wr.release()
cv2.destroyAllWindows() ##destroy all windows