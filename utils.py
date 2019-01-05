import numpy as np
from PIL import Image


def toImg(npArray, mode = "RGB"):
	return Image.fromarray(np.uint8(npArray), mode = mode)

def customName(init, db):
	i = 0
	while init + str(i) in db:
		i += 1
	else:
		return init + str(i)