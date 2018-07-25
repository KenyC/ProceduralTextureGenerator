from operators.functions.perlin import *
from op import Op
from cst import *

class Perlin(Op):
	def __init__(self, detail = D_DETAIL, size = D_SIZE, seed = None):
		super(Perlin, self).__init__(perlin, size =  size, detail = detail, seed = seed)

class GridPerlin(Op):
	def __init__(self, detail = D_DETAIL, size = D_SIZE, clampGrad = D_CLAMP_GRAD, seed = None):
		super(GridPerlin, self).__init__(gridNoise, size =  size, detail = detail, seed = seed)