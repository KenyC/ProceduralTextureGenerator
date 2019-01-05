from PIL import Image, ImageDraw
from PIL.ImageChops import add
from functools import reduce
import numpy as np

PRECISION_CRACKS = 2.
"""
Creates cracks
Args
	- thick: thickness of lines
	- N: number of lines
	- mean/sigL: parameters of the length distribution
	- sigD: variation of direction of random movement (rad.unit-1)
Lengths measurement are in percent of image width
"""
def crack(size, thick, N, meanL, sigD, sigL, seed = None, mode = "RGB"):
	
	width, height = size
	if seed is not None:
		np.random.seed(seed)

		
	# Image is bigger, will be cropped later on for wrapping purposes
	img = Image.new(mode, (3 * width, 3 * height), (0,0,0))
	draw = ImageDraw.Draw(img)

	shapeL, scaleL = (meanL/sigL)**2, sigL**2/meanL

	# Draw origins of line in the middle image
	origins = np.stack((np.random.randint(width, 2*width, N), np.random.randint(height, 2 * height, N)), axis = -1)
	
	# Draw lengths
	lengths = np.random.gamma(shapeL, scaleL, N)

	# Direction variance fit to precision
	trueSigD = sigD * PRECISION_CRACKS / width

	for i in range(N):

		# number of points in line
		n = int(lengths[i] * width / PRECISION_CRACKS)
		
		# Draw initial direction
		initDir = np.random.uniform(0.0, 2*np.pi)

		# Initialize dirs array
		dirs = np.full(n, initDir, dtype = "float")

		# Initialize noise for direction
		noise = np.cumsum(np.random.normal(0., trueSigD, n))
		#print(noise[5])
		dirs += noise

		# Vectorialize direction
		vDirs = np.stack((np.cos(dirs), np.sin(dirs)), axis = -1)
		

		# Create poly line
		polyL = np.resize(origins[i], (n,2)) + np.cumsum(vDirs, axis = 0)
		polyL = [(int(p[0]),int(p[1])) for p in polyL]

		# Draw lines
		draw.line(polyL, width = thick, fill = (255,255,255))

	cropImgs = [img.crop((i*width,j*height,(i+1)*width,(j+1)*height)) for i in range(3) for j in range(3)]

	return reduce(add, cropImgs)

		