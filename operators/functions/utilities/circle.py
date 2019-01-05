from PIL import Image, ImageDraw
import numpy as np


D_COL = (0, 0, 0)

"""
Draws a circle in the middle of surface filled with uniform colour
Args
	- colour: a tuple of colour components (size should be as in mode)
"""
def circle(size, radius, colour, bgColour = D_COL, mode = "RGB"):

	width, height = size
	img = Image.new(mode, size, D_COL)
	draw = ImageDraw.Draw(img)

	boundingBox = width/2 - radius, height/2 - radius, width/2 + radius, height/2 + radius

	draw.ellipse(boundingBox, fill = colour)
	del draw

	return img


"""
Draws a RGB circle of random colour
"""
def randomCircle(size, radius, bgColour = D_COL, seed = None, mode = "RGB"):
	if seed is not None:
		np.random.seed(seed)

	return circle(size, radius, tuple(x for x in np.random.randint(0, 256, 3)), bgColour)
