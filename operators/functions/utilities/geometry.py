import numpy as np
from math import sqrt

def fromAbsToRel(point, origin, relToAbsMatrix):
	absToRel = np.linalg.inv(relToAbsMatrix)

	return absToRel.dot(point - origin)

# In a periodic lattice, find instance of point right in upper right quadrant of origin
def findRBRepr(point, origin, relToAbs):
	absToRel = np.linalg.inv(relToAbs)

	closestIntegerPt = absToRel.dot(point - origin) % 1

	return relToAbs.dot(closestIntegerPt) + origin



# In a periodic lattice, find instance of point closest to origin
# def findClosestRepr(point, origin, relToAbs):
# 	absToRel = np.linalg.inv(relToAbs)

# 	closestIntegerPt = np.round(absToRel.dot(point - origin))
# 	return relToAbs.dot(closestIntegerPt) + origin

def findClosestRepr(point, origin, relToAbs):
	bot = findRBRepr(point, origin, relToAbs)
	l = np.stack([bot, bot - relToAbs[:, 0], bot - relToAbs[:, 1], bot - relToAbs[:, 0] - relToAbs[:, 1]])

	ds = np.linalg.norm(l - origin, axis = 1)
	return l[np.argmin(ds),:]


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

def two_by_two(iterator):
	try:
		before = next(iterator, None)
	except StopIteration:
		return

	while True:
		try:
			after = next(iterator)
		except StopIteration:
			return
		yield before, after
		before = after


def is_convex(pts, intolerance = 1e-5):

	path = [v2 - v1 for v1, v2 in two_by_two(iter(list(pts) + [pts[0]]))]
	path += [path[0]]

	vectProd = [np.cross(v1, v2) for v1, v2 in two_by_two(iter(path))]

	return (all(x >=0 for x in vectProd) or all(x <=0 for x in vectProd)) and not any(abs(x) < intolerance for x in vectProd)


def ptInTriangle(point, triangle, intolerance = 0):

	origin = triangle[0]
	axis1 = triangle[1] - triangle[0]
	axis2 = triangle[2] - triangle[0]

	matrix = np.stack([axis1, axis2]).T
	invMat = np.linalg.inv(matrix)

	solution = invMat.dot(point - origin)
	# print(solution)

	return solution[0] >= intolerance and solution[1] >= intolerance and solution[0] + solution[1] <= 1 - intolerance



if __name__ == "__main__":

	mat  = np.array([[1.0, 0.0], [0.5, sqrt(3.)/2.]]).T
	mat2 = np.linalg.inv(mat)
	a = np.array([1.3, 0.8])
	b = np.array([1.58, 2.1])
	c = np.array([3, 2.])
	d = np.array([3, 4.])

	pts = np.stack([a, b, c, d], axis = 0)
