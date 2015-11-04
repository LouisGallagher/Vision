import cv2, numpy
from matplotlib import pyplot as plt

img = cv2.imread('Data/starrySky.jpg', cv2.IMREAD_COLOR)  #loads an color image from storage

cv2.namedWindow('Night Sky', cv2.WINDOW_NORMAL)
cv2.imshow('Night Sky', img) #load the image into a named window with the given title

k = cv2.waitKey(0) 

if k == 27:		##if esc key pressed
	cv2.destroyAllWindows()  ##destroy all open windows 
elif k== ord('s'):		#else if s key pressed save then exit
	cv2.imwrite('nightsky.png', img)
	cv2.destroyAllWindows()
