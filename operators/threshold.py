from operators.functions.threshold import *
from operators.functions.flat import *
from op import Op
from cst import *

class Threshold(Op):
	def __init__(self, img = flat(), thres = D_THRES, seed = None):
		super(Threshold, self).__init__(threshold, img = img, thres = thres)
