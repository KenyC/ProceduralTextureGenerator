from operators.functions.cracks import *
from operators.functions.flat import *
from op import Op
from cst import *

class Cracks(Op):
	def __init__(self, thick, N, meanL, sigD, sigL, seed = None, mode = "RGB"):
		super(Threshold, self).__init__(crack, thick, N, meanL, sigD, sigL, seed, mode)
