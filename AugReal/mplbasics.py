import numpy, cv2
from matplotlib import pyplot as plt

img = cv2.imread('Data/starrysky.jpg', 0)  #read image from memory
plt.imshow(img, cmap ='gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])
plt.show()