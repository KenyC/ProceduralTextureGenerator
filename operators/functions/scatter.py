from PIL import Image, ImageDraw
from PIL.ImageChops import lighter
from functools import reduce
from cst import *
from utils import toImg
import numpy as np



"""
Scatters samples from distribution across surface
Args
	- distribution: a function taking a seed as sole (optional) argument, and returning one sample
	- N: number of samples
	- uniform: from 0 to 1, determines how homogeneous the distribution of points will be
"""
def scatter(size, distribution, N, uniform = 0.5, seed = None):
	
	width, height = size
	if seed is not None:
			np.random.seed(seed)

	# toReturn = Image.new("RGBA", (3 * width, 3 * height), (0, 0, 0, 0))
	rgbSum = np.full((width, height, 3), (0.,0.,0.), dtype = "float")
	alphaSum = np.full((width, height), 0.0, dtype = "float")
	alphaMax = np.full((width, height), 0.0, dtype = "float")
	

	origins = np.stack((np.random.randint(width, 2*width, N), np.random.randint(height, 2 * height, N)), axis = -1)


	for i in range(N):

		orig = origins[i]

		if seed is None:
			img = distribution()
		else:
			img = distribution(seed + i)

		boundingBox = orig[0] - img.width // 2, orig[1] - img.height // 2, orig[0] + img.width // 2, orig[1] + img.height // 2
		toPaste = Image.new("RGBA", (3 * width, 3 * height), (0, 0, 0, 0))
		toPaste.paste(img, boundingBox)
		cropImgs = [toPaste.crop((i*width,j*height,(i+1)*width,(j+1)*height)) for i in range(3) for j in range(3)]
		toPaste = reduce(lighter, cropImgs)

		toPaste = np.array(toPaste, dtype = "float")

		rgb, alpha = toPaste[:,:,:-1], toPaste[:,:,-1]

		print(alphaSum.shape)
		rgbSum += (1/N) * rgb * alpha[..., np.newaxis]
		alphaSum += (1/N) * alpha
		alphaMax = np.maximum(alphaMax, alpha)

	

	return toImg(np.concatenate((rgbSum/alphaSum[..., np.newaxis],alphaMax.reshape(*alphaMax.shape,1)), axis = -1), "RGBA")
