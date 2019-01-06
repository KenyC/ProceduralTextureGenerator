from PIL import Image, ImageDraw
from PIL.ImageChops import lighter
from functools import reduce
#from cst import *
#from utils import toImg
import numpy as np
from voronoi_utils_new import VoronoiGraph







def voronoiRandom(size, N, width, seed = None):
	height, width = size

	if seed is not None:
		np.random.seed(seed)

	# Draw points in the middle image
	points = np.stack((np.random.randint(0, width, N), np.random.randint(0, height, N)), axis = -1).astype("float")
	print(points)

	decalX = np.array([width, 0.], dtype = "float")
	decalY = np.array([0., height], dtype = "float")
	points = np.concatenate([points + i*decalX + j*decalY for i in range(3) for j in range(3)])
	points = points + np.random.normal(scale = 1/N, size = points.shape)

	voronoi = VoronoiGraph(points.tolist())

	return voronoi.draw_img((3 * width, 3 * height)).crop((width, height, 2*width, 2*height))

def voronoiFromPts(size, points, width):
	pass

if __name__ == "__main__":
	N = 20
	size = height, width = 300 ,300
	pts = np.stack((np.random.randint(0, 300, N), np.random.randint(0, 300, N)), axis = -1)
	# Jitter to avoid degenerates
	pts = pts +  np.random.normal(scale = 1/N, size = pts.shape)

	voronoi = VoronoiGraph(pts.tolist())

	img =  voronoi.draw_img(size)
	