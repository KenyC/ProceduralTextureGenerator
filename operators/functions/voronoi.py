from PIL import Image, ImageDraw
from PIL.ImageChops import lighter
from functools import reduce
#from cst import *
#from utils import toImg
import numpy as np
from utilities.voronoi_utils import VoronoiGraph






def voronoiFromPts(size, points, thick = 1.):
	width, height = size

	# Create 9 copies of each points for tiling
	decalX = np.array([width, 0.], dtype = "float")
	decalY = np.array([0., height], dtype = "float")
	points = np.concatenate([points + i*decalX + j*decalY for i in range(3) for j in range(3)])

	# Jitter to avoid degenerate cases
	points = points + np.random.normal(scale = 1/100., size = points.shape)

	voronoi = VoronoiGraph(points.tolist())

	return voronoi.draw_img(size, center = (width, height), thick = thick)

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


	return voronoiFromPts(size, points, thick)





if __name__ == "__main__":
	N = 20
	size = height, width = 300 ,300
	pts = np.stack((np.random.randint(0, 300, N), np.random.randint(0, 300, N)), axis = -1)
	# Jitter to avoid degenerates
	pts = pts +  np.random.normal(scale = 1/N, size = pts.shape)

	voronoi = VoronoiGraph(pts.tolist())

	img =  voronoi.draw_img(size)
	