from PIL import Image, ImageDraw
from PIL.ImageChops import lighter
from functools import reduce
#from cst import *
#from utils import toImg
import numpy as np
from math import sqrt

from utilities.voronoi_utils import VoronoiGraph
from utilities.pt_dist import triangular_range, jitter






def voronoiFromPts(size, points, thick = 1., fill = False):
	width, height = size

	# Create 9 copies of each points for tiling
	decalX = np.array([width, 0.], dtype = "float")
	decalY = np.array([0., height], dtype = "float")
	points = np.concatenate([points + i*decalX + j*decalY for i in range(3) for j in range(3)])

	# Jitter to avoid degenerate cases
	points = points + np.random.normal(scale = 1/100., size = points.shape)

	voronoi = VoronoiGraph(points.tolist())
	if not fill:
		return voronoi.draw_img(size, center = (width, height), thick = thick)
	else:
		return voronoi.fill_img((width, height), center = (width, height))

def voronoiPureRandom(size, N, thick = 1., seed = None):
	width, height = size

	if seed is not None:
		np.random.seed(seed)

	# Draw points in the middle image
	points = np.stack((np.random.randint(0, width, N), np.random.randint(0, height, N)), axis = -1).astype("float")

	return voronoiFromPts(size, points, thick)

"""

	- uniform is a parameter from 0 to infty 0 ; means evenly distributed, infty means random
	it represents the amount of jitter in percentage of the dimensions of the picture so 100 is for all intent and purposes, compeltely random
"""
def voronoiRandom(size, sqNM, uniform = 5., thick = 1, seed = None):

	if isinstance(sqNM, int):
		N, M = sqNM, sqNM
	else:
		N, M = sqNM

	if isinstance(size, int):
		width, height = size = size, size
	else:
		width, height = size

	if seed is not None:
		np.random.seed(seed)

	spX = width / (N + 1)
	spY = height / (M + 1)

	coordX, coordY = np.meshgrid(np.linspace(spX, width - spX, N), np.linspace(spY, height - spY, M))

	coordX = (coordX + np.random.normal(scale = uniform/100. * width, size = coordX.shape)) % width
	coordY = coordY + np.random.normal(scale = uniform/100. * height, size = coordY.shape) % height

	points = np.stack((coordX, coordY), axis = -1).reshape((N * M, 2))

	print("FREAKING NPTS", points.shape, N, M)

	return voronoiFromPts(size, points, thick)

# scale is proportional to shortest neighbour ;
# 0.1 to 0.2 seems to be the esthetically pleasing range
def voronoiTriangular(width, n_x, scale, thick = 1, fill = False):
	step_x = width / n_x
	step_y = step_x * sqrt(3)/2

	height = int(2 * step_y * (width // (2 * step_y)))
	n_y = int(height // step_y) + 1

	pts = triangular_range(n_x, n_y, step_x)
	pts = jitter(pts, scale * step_x)

	return voronoiFromPts((width, height), pts, thick, fill)




if __name__ == "__main__":
	N = 20
	size = height, width = 300 ,300
	pts = np.stack((np.random.randint(0, 300, N), np.random.randint(0, 300, N)), axis = -1)
	# Jitter to avoid degenerates
	pts = pts +  np.random.normal(scale = 1/N, size = pts.shape)

	voronoi = VoronoiGraph(pts.tolist())

	img =  voronoi.draw_img(size)
	