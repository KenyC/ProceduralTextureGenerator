from PIL import Image
from functools import reduce
import numpy as np
from utils import toImg



def multiresolution(**d):

	def toOp(i):
		return d["o" + str(i)]

	def toCoeff(i):
		return d["c" + str(i)]
	
	coeffs = []
	imgs = []
	n = 0
	while ("o"+ str(n) in d) and ("c" + str(n) in d):
		coeffs.append(toCoeff(n))
		imgs.append(toOp(n))
		n += 1

	coeffs = np.array(coeffs)
	coeffs = coeffs / np.sum(coeffs)

	return toImg(sum(coeffs[i] * imgs[i] for i in range(n)))