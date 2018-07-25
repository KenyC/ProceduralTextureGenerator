from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image

active = None
img_test = Image.open("index.jpg","r")

def display(img):
	imshow(np.asarray(img))
