from PIL import Image
from cst import *

def flat(color = D_COL, size = D_SIZE, mode = D_MODE):
	"""Returns an image with uniform color"""
	return Image.new(mode, size, color)