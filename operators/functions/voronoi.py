from PIL import Image, ImageDraw
from PIL.ImageChops import lighter
from functools import reduce
from cst import *
from utils import toImg
import numpy as np






def voronoiRandom(size, scale, width, seed = None):
	
	if seed is not None:
		np.random.seed(seed)



	return voronoiFromPts(size, points, width)

def voronoiFromPts(size, points, width):
	pass