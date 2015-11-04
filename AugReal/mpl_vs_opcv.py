## To demonstrate the different image representations used by
## OpenCV and Matplotlib
## Namely OpenCV uses BGR ordering while mpl uses RGB ordering

import cv2, numpy
from matplotlib import pyplot as plt

img = cv2.imread('Data/starrySky.jpg')

img2 = img[:,:,::-1]

plt.subplot(121);plt.imshow(img)
plt.subplot(122);plt.imshow(img2)
plt.show()

cv2.imshow('bgr image', img)
cv2.imshow('rgb image', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
