import cv2, numpy as np 
import math as m
import matplotlib.pyplot as plt

#-----------Section: Rigid body transformations-----------#

# function: computes a rotation matrix based on the euler angles of
# a rigid body.
#
# parameters:
# alpha: angle to rotate about the x axis 
# beta: angle to rotate about the y axis
# gamma: angle to rotate about the z axis
def eulerToR(alpha, beta, gamma):
		
	#alphaRad, betaRad, gammaRad = np.radians(([alpha, beta, gamma])) # convert degrees to radians
	
	## perform trig functions
	cosAlpha = m.cos(alpha)
	sinAlpha = m.sin(alpha)
	cosBeta  = m.cos(beta)
	sinBeta  = m.sin(beta)
	cosGamma = m.cos(gamma)
	sinGamma = m.sin(gamma)
	
	# compute rotation matrices for rotating about x, y and z axes 
	rx = np.array([[1, 0, 0], [0, cosAlpha, -1 * sinAlpha], [0, sinAlpha, cosAlpha]])  
	ry = np.array([[cosBeta, 0, sinBeta], [0, 1, 0], [-1 * sinBeta, 0, cosBeta]])      
	rz = np.array([[cosGamma, -1 * sinGamma, 0], [sinGamma, cosGamma, 0], [0, 0, 1]])  
	
	return np.dot(rz, np.dot(ry, rx))  # multiply together for roll, pitch and yaw


# function: computes a rotation matrix based on Rodriguez formula 
#
# parameters:  w1, w2, w3 represents the vector about which the rotation is performed
# we have ||(w1, w2, w3)|| = theta 
# where theta is the angle of the rotation
def expToR(w1, w2, w3):		
	identity = np.identity(3)													   # 3 space identity matrix
	wHat = np.array([[0, -1 * w3 , w2], [w3, 0, -1 * w1], [-1 * w2, w1, 0]])
	wMagnitude = np.linalg.norm(np.array([[w1, w2, w3]]))   					   # the magnitude of the vector w
		
	return identity + ((wHat / wMagnitude) * m.sin(m.radians(wMagnitude))) + \
	((np.dot(wHat, wHat) / pow(wMagnitude, 2)) * (1 - m.cos(m.radians(wMagnitude))))	# Rodriguez formula for computing rotation matrix





# function: computes a transformation matrix in homogeneous form from the 6D pose passed in as parameters, where R, the rotation
# component, is computed using euler angles.
#
# parameters:
# X, Y, Z is the point in the world coordinate system 
# alpha, beta, gamma;  the angles of rotation about the x, y and z axes respectively (see funtion eulerToR for more information)
def eulerToT(X, Y, Z, alpha, beta, gamma):
	t = np.array([[X, Y, Z]]).T
	r = eulerToR(alpha, beta, gamma)
	T = np.append(r, t, 1)
	T = np.append(T,np.array([[0, 0, 0, 1]]), 0)

	return T




# function: computes a transformation matrix in homogeneous form from the 6D pose passed in as parameters, where R, the rotation
# component, is computed using Rodriguez formula and exponential coordinates.
#
# parameters:
# X, Y, Z is the point in the world coordinate system 
# w1, w2, w3 : the vector about which the rotation is performed (see function expToR for more details)
def expToT(X, Y, Z, w1, w2, w3):
	t = np.array([[X, Y, Z]]).T
	r = expToR(w1, w2, w3)
	T = np.append(r, t, 1)
	T = np.append(T,np.array([[0, 0, 0, 1]]), 0)

	return T

#----------------------end rigid body transformations-----------------------------

#----------------------section: Camera matrix--------------------------------------

# function takes the intrinsic parameters of a camera and returns a perspective projection matrix for the camera in homogeneous form 
#
# parameters:
# f - the focal length of the camera
# k,l - scale factor relating retinal to image coordinates
# u0, v0 - represent the principal point 

def intrinsicsToK(f, k, l, u0, v0):
		return np.array([[f * k, 0, u0 , 0], [0, f * l, v0 , 0], [0, 0, 1, 0]])


# def simulateCamera(f, h, d, ch):
# 	imagePlane = np.array([[[255 for i in range(3)] for x in range(h)] for y in range(d)], dtype = np.uint8)
# 	K  = intrinsicsToK(f, 1, 1,  int(h/2), int(d/2) ) ##something not right with h and d  
# 	#K = np.delete(K, (-1), axis = 1)
# 	T = eulerToT(0, 0 , ch, 0, 90, -90)
# 	#T = np.delete(T, (-1), axis = 0)
# 	M = np.dot(K, T)
# 	#M = np.delete(M, (2), axis = 1)
	
	
# 	for i in range(1000):
# 		pl1Image = np.dot(M, np.array([[i,1,0,1]]).T)
# 		pl2Image = np.dot(M, np.array([[i,-1,0,1]]).T)
# 		pl3Image = np.dot(M, np.array([[i,0,0,1]]).T)
# 		x1im, y1im = [int(x / pl1Image[2]) for x in pl1Image[:2]]
# 		x2im, y2im = [int(x / pl2Image[2]) for x in pl2Image[:2]]
# 		x3im, y3im = [int(x / pl3Image[2]) for x in pl3Image[:2]]

# 		if 0 <= y1im < h and 0<= x1im < d:	 		
# 			imagePlane[y1im , x1im] = [255,0,0]

# 		if 0 <= y2im < h and 0 <= x2im< d:	 		
# 			imagePlane[y2im , x2im] = [0,0,255]

# 		if 0 <= y3im < h and 0 <= x3im< d:	 		
# 			imagePlane[y3im , x3im] = [0,0,255]	
	
# 	return imagePlane



def simulateCamera():
	T = eulerToT(0, 0, -20, 0, np.pi/2, -1 * (np.pi/2))	
	K = intrinsicsToK(500, 1, 1, 0, 0)
	M = K.dot(T)
	M = np.delete(M, (2), axis = 1)
	
	fig = plt.figure()
	ax = fig.gca() 

	for i in range(100):
		p1 = M.dot(np.array([[i, 1, 1]]).T)
		p2 = M.dot(np.array([[i, 0, 1]]).T)
		p3 = M.dot(np.array([[i, -1, 1]]).T)
		ax.plot(p1[0]/p1[2],p1[1]/p1[2],'r.')
		ax.plot(p2[0]/p2[2],p2[1]/p2[2],'k.')
		ax.plot(p3[0]/p3[2],p3[1]/p3[2],'g.')

	
	

	# # e1 = np1.hstack(np.array([[i, 1, 1]]).T for i in range(10))
	# # e2 = np.hstack(np.array([[i, 0, 1]]).T for i in range(10))
	# # e3 = np.hstack(np.array([[i, -1, 1]]).T for i in range(10))

	
	# i1 = M.dot(e1)
	# i2 = M.dot(e2)
	# i3 = M.dot(e3)


	
	# ax.plot(p1[0]/p1[2],p1[1]/p1[2],'r.')
	plt.show()

