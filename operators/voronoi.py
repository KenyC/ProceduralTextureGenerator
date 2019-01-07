from operators.functions.voronoi import *
from op import Op
from cst import *



class PureVoronoi(Op):
	def __init__(self, N, thick = 1., seed = None)
		super(PureVoronoi, self).__init__(voronoiPureRandom, N = N, thick = 1., seed = None)


class Voronoi(Op):
	def __init__(self, dsqNM, uniform = 5., thick = 1, seed = None):
		super(Voronoi, self).__init__(voronoiRandom, dsqNM = dsqNM, uniform = 5., thick = 1, seed = None):