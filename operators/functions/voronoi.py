from PIL import Image, ImageDraw
from PIL.ImageChops import lighter
from functools import reduce
#from cst import *
#from utils import toImg
import numpy as np
from voronoi_utils import VoronoiGraph







def voronoiRandom(size, N, width, seed = None):
	height, width = size

	if seed is not None:
		np.random.seed(seed)

	# Draw points in the middle image
	points = np.stack((np.random.randint(0, width, N), np.random.randint(0, height, N)), axis = -1).astype("float")
	print(points)

	voronoi = VoronoiGraph(points.tolist())

	return voronoi.draw_img(size), points

def voronoiFromPts(size, points, width):
	pass

if __name__ == "__main__":
	N = 20
	size = height, width = 300 ,300
	pts = np.stack((np.random.randint(0, 300, N), np.random.randint(0, 300, N)), axis = -1)
	# Jitter to avoid degenerates
	pts = pts +  np.random.normal(scale = 0.1, size = pts.shape)

	voronoi = VoronoiGraph(pts.tolist())

	img =  voronoi.draw_img(size)
	