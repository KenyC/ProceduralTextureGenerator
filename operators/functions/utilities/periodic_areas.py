



class Area:

	def __init__(self, pts = None, edges = None):
		self.pts = pts # iterable of RVertex
		self.edges = edges # iterable Edge

	def connects(self, *pts):
		return all(any(pt2 == pt1 for pt2 in self.pts) for pt1 in pts)


class RArea:

	def __init__(self, area, offset, mat):
		self.base = area # Area
		self.offset = offset # array(2)
		self.mat = mat # array(2,2)

	def connects(self, *pts):
		return self.base.connects(*pts)
		
