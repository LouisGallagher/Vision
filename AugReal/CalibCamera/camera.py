import cv2, numpy as np, math as m


#-----------Section: Rigid body transformations-----------#

# function: computes a rotation matrix based on the euler angles of
# a rigid body.
#
# parameters:
# alpha: angle to rotate about the x axis 
# beta: angle to rotate about the y axis
# gamma: angle to rotate about the z axis
def eulerToR(alpha, beta, gamma):
		
	alphaRad, betaRad, gammaRad = np.radians(([alpha, beta, gamma])) # convert degrees to radians
	
	## perform trig functions
	cosAlpha = m.cos(alphaRad)
	sinAlpha = m.sin(alphaRad)
	cosBeta  = m.cos(betaRad)
	sinBeta  = m.sin(betaRad)
	cosGamma = m.cos(gammaRad)
	sinGamma = m.sin(gammaRad)
	
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
	wMagnitude = np.linalg.norm(np.array([[w1, w2, w3]]))   
		
	return identity + ((wHat / wMagnitude) * m.sin(m.radians(wMagnitude))) + \
	((np.dot(wHat, wHat) / pow(wMagnitude, 2)) * (1 - m.cos(m.radians(wMagnitude))))





# function: computes a transformation matrix in homogeneous form from the 6D pose passed in as parameters, where R, the rotation
# component, is computed using euler angles.
#
# parameters:
# X, Y, Z is the point in the world coordinate system 
# alpha, beta, gamma;  the angles of rotation about the x, y and z axes respectively (see funtion eulerToR for more information)
def eulerToT(X, Y, Z, alpha, beta, gamma):
	pwrl = np.array([[X, Y, Z]]).T
	r = eulerToR(alpha, beta, gamma)
	T = np.append(r, pwrl, 1)
	T = np.append(T,np.array([[0, 0, 0, 1]]), 0)

	return T




# function: computes a transformation matrix in homogeneous form from the 6D pose passed in as parameters, where R, the rotation
# component, is computed using Rodriguez formula and exponential coordinates.
#
# parameters:
# X, Y, Z is the point in the world coordinate system 
# w1, w2, w3 : the vector about which the rotation is performed (see function expToR for more details)
def expToT(X, Y, Z, w1, w2, w3):
	pwrl = np.array([[X, Y, Z]]).T
	r = expToR(w1, w2, w3)
	T = np.append(r, pwrl, 1)
	T = np.append(T,np.array([[0, 0, 0, 1]]), 0)

	return T