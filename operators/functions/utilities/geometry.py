import numpy as np
from math import sqrt

def fromAbsToRel(point, origin, relToAbsMatrix):
	absToRel = np.linalg.inv(relToAbsMatrix)

	return absToRel.dot(point - origin)

# In a periodic lattice, find instance of point right in upper right quadrant of origin
def findRBRepr(point, origin, relToAbs):
	absToRel = np.linalg.inv(relToAbs)

	closestIntegerPt = absToRel.dot(point - origin) % 1
	assert(closestIntegerPt[0] >= 0 and closestIntegerPt[1] >= 0)

	return relToAbs.dot(closestIntegerPt) + origin

# In a periodic lattice, find instance of origin closest to point
def findClosestRepr(point, origin, relToAbs):
	absToRel = np.linalg.inv(relToAbs)

	closestIntegerPt = np.round(absToRel.dot(point - origin))
	return relToAbs.dot(closestIntegerPt) + origin
	# bot = findRBRepr(point, origin, relToAbs)
	# l = np.stack([bot, bot - relToAbs[:, 0], bot - relToAbs[:, 1], bot - relToAbs[:, 0] - relToAbs[:, 1]])

	# ds = np.linalg.norm(l - point, axis = 0)

	# b, _ = min(zip(l, ds), key = lambda x: x[1])
	# return b 


"""
Returns a tuple containing absolute coordinate of 
"""
def findBoundingBox(points, origin, relToAbs):
	absToRel = np.linalg.inv(relToAbs)

	rightBelowPt = np.floor(absToRel.dot((points - origin).T))
	rightAbovePt = np.ceil(absToRel.dot((points - origin).T))

	minPts = np.min(rightBelowPt, axis = 1)
	maxPts = np.max(rightAbovePt, axis = 1)

	return relToAbs.dot(minPts) + origin, maxPts - minPts



if __name__ == "__main__":

	mat  = np.array([[1.0, 0.0], [0.5, sqrt(3.)/2.]]).T
	a = np.array([1.3, 0.8])
	b = np.array([1.58, 2.1])
	c = np.array([3, 2.])
	d = np.array([3, 4.])

	pts = np.stack([a, b, c, d], axis = 0)
