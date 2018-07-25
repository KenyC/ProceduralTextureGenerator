from matplotlib.pyplot import imshow
import numpy as np
from math import floor, ceil
from PIL import Image
from PIL.ImageChops import blend
from cst import *

# Linear interpolation between values
def lerp(x, y, w):
	return (1.-w)*x+w*y

# Hermite interpolation between values
def hermit(x, y, w):
	return x + (y-x) * (3*w-2)*(w**2) 

# Quintic interpolation between values
def quintic(x, y, w):
	return x + (y-x) * ((6*w-15)*w+10)*w**3 


# Function for creating gray-scale Perlin Noise
def perlin(size, detail, seed = None, interpolation = quintic):

	width, height = size 

	if seed is not None:
		np.random.seed(seed)

	gradVect = np.random.normal(size = (detail,detail,2))
	norms = np.linalg.norm(gradVect, axis=-1)
	gradVect = gradVect / norms[..., np.newaxis]

	img = np.full((width,height,3), 0.0, dtype = "float")
	vOnes = np.ones(3)

	for i in range(width):
		for j in range(height):

			
			# Compute actual position in grid space
			x, y = (i + 0.5)/width*(detail), (j + 0.5)/height*(detail)

			# compute indices of upper left corner of cell
			xD, yD = floor(x), floor(y)

			# Compute coordinates in cell space
			sx = x % 1.
			sy = y % 1.

			# Compute grid points gradient
			v00 = np.dot(gradVect[xD,yD],[sx, sy]) # upper left
			v10 = np.dot(gradVect[(xD + 1) % detail, yD],[sx - 1, sy]) # upper right
			v01 = np.dot(gradVect[xD, (yD + 1) % detail],[sx, sy - 1]) # lower left
			v11 = np.dot(gradVect[(xD + 1) % detail, (yD + 1) % detail],[sx - 1, sy- 1]) # lower right

			# interpolate
			vUp = interpolation(v00, v10, sx)
			vDown = interpolation(v01, v11, sx)
			img[i,j] = (interpolation(vUp, vDown, sy) * vOnes + 1.)/2.
	
	imshow(img)
	return Image.fromarray(np.uint8(np.transpose(img,(1,0,2))*255))

# Perlin noise in 1D, in the direction provided by horiz
def perlin1D(size, detail, horiz = True, seed = None, clampGrad = D_CLAMP_GRAD, interpolation = quintic):

	width, height = size 
	mainDim = width if horiz else height
	secondDim = height if horiz else width

	if seed is not None:
		np.random.seed(seed)

	imgsPt = np.random.uniform(-clampGrad,clampGrad,size = detail)
	

	img = np.full((mainDim,3), 0.0, dtype = "float")
	vOnes = np.ones(3)

	for i in range(mainDim):
		# position in grid coordinate
		iGrid = ((i+0.5) / mainDim) * detail
		# position in cell
		s = iGrid % 1 

		
		img[i] = (interpolation(imgsPt[floor(iGrid)] * s, imgsPt[ceil(iGrid) % detail] * (s-1), s) + 1.) / 2. * vOnes
		
	
	img = np.tile(img, (secondDim,1,1))
	if horiz:
		img = np.transpose(img, (1,0,2))
	
	return Image.fromarray(np.uint8(np.transpose(img,(1,0,2))*255))

# Anistropic noise, grid-like, obtained by superposing two 1-D Perlin noise
def gridNoise(size, detail, seed = None, clampGrad = D_CLAMP_GRAD,  interpolation = quintic):

	# Horizontal noise
	imgH = perlin1D(size, detail, True, seed, clampGrad, interpolation)

	# Vertical noise
	imgV = perlin1D(size, detail, False, seed + 1 if seed is not None else None, clampGrad, interpolation)


	return blend(imgH, imgV, 0.5)