import numpy as np
from PIL import Image


def toImg(npArray):
	return Image.fromarray(np.uint8(npArray))

def customName(init, db):
	i = 0
	while init + str(i) in db:
		i += 1
	else:
		return init + str(i)