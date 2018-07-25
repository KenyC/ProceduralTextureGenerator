from operators.functions.multiresolution import multiresolution
from op import Op
from cst import *

class Multiresolution(Op):
	def __init__(self, **d):
		super(Multiresolution, self).__init__(multiresolution, **d)
