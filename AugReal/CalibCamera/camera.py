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
		
	alpharad, betarad, gammarad = np.radians(([alpha, beta, gamma])) # convert degrees to radians
	
	## perform trig functions
	cosalpha = m.cos(alpharad)
	sinalpha = m.sin(alpharad)
	cosbeta  = m.cos(betarad)
	sinbeta  = m.sin(betarad)
	cosgamma = m.cos(gammarad)
	singamma = m.sin(gammarad)
		
	rx = np.array([[1.0, 0.0, 0.0], [0.0, cosalpha, -1.0 * sinalpha], [0.0, sinalpha, cosalpha]])
	ry = np.array([[cosbeta, 0.0, sinbeta], [0.0, 1.0, 0.0], [-1.0 * sinbeta, 0.0, cosbeta]])
	rz = np.array([[cosgamma, -1.0 * singamma, 0.0], [singamma, cosgamma, 0.0], [0.0, 0.0, 1.0]])
	
	return np.dot(rz, np.dot(ry, rx)) 
	