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


def simulateCamera(f, k, l, u0, v0, h):
	T = eulerToT(0, 0, h, 0, np.pi/2, -1 * (np.pi/2))	
	K = intrinsicsToK(f, k, l, u0, v0)
	M = K.dot(T)
	M = np.delete(M, (2), axis = 1)
	fig = plt.figure()
	ax = fig.gca() 

	l1 = M.dot(np.hstack([np.array([[i,1,1]]).T for i in range(1000)]))
	l2 = M.dot(np.hstack([np.array([[i,0,1]]).T for i in range(1000)]))
	l3 = M.dot(np.hstack([np.array([[i,-1,1]]).T for i in range(1000)]))
		
	ax.plot(l1[0,:] / l1[2,:], l1[1,:] /l1[2,:], 'r.')
	ax.plot(l2[0,:] / l2[2,:], l2[1,:] /l2[2,:], 'b.')
	ax.plot(l3[0,:] / l3[2,:], l3[1,:] /l3[2,:], 'g.')

	plt.show()

def calibrateCamera3D(data):
	data = np.loadtxt(data)
	Pwrl = np.array(data[:, :3])
	Pim = np.array(data[:, 3:])
	constraintMatrix  = np.array([[]])

	# convert im points to homogeneuous coordinates 
	col = np.array([[1 for x in range(len(Pim))]]).T
	Pim = np.append(Pim,col,1) 
	
	# add the constraints
	for i in range(len(Pwrl)):
		constraint = np.array([[Pwrl[:, :], 0, ] , [0, Pwrl[:, :] ]])
		np.append(constraints, constraint, 0)

	# solve for x using linear least squares   
	M = numpy.linalg.lstsq(constraints, np.array[[0 for x in range()]].T)