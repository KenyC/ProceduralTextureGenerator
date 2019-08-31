from component import Vertex, Edge, Area
import numpy as np



class Vertex:

	def __init__(self, coords, links = [], pos = None):
		self.coords = coords
		self.links = links
		self.pos = None

	def add_link(self, other, edge, isOrigin):
		self.links.append(Link(other, edge, isOrigin))

	def remove_link(self, other):
		for i, l in enumerate(self.links):
			if l.other is other:
				del self.links[i]

	def r(self, relToAbs, rep = np.zeros(2)):
		return RVertex(self, rep.copy(), relToAbs)


class RVertex:
	
	def __init__(self, vertex, rep, relToAbs):
		self.vertex = vertex
		self.rep = rep
		self.mat = relToAbs

	@property
	def force(self):
		return self.vertex.force

	@force.setter
	def force(self, value):
		self.vertex.force = value
	
	

	@property
	def coords(self):
		return self.vertex.coords + self.mat.dot(self.rep)

	@coords.setter
	def coords(self, value):
		self.vertex.coords = value - self.mat.dot(self.rep)

	def neighbours(self):
		for link in self.vertex.links:
			yield RVertex(vertex = link.other, rep = self.rep + link.offset, relToAbs = self.relToAbs)

	def __eq__(self, other):
		return (self.vertex is other.vertex) and np.all(self.rep == other.rep)

	def offset(self, rep1):
		return RVertex(self.vertex, self.rep + rep1, self.mat)

	

class Edge:
	def __init__(self, origin, end, offset, pos = None):
		self.origin = origin # RVertex
		self.end = end # RVertex
		self.offset = offset # array(2)
		self.pos = pos # DList

	def link(rv1, rv2):
		e = Edge(origin = rv1, end = rv2, offset = rv2.rep - rv1.rep)
		rv1.vertex.add_link(rv2.vertex, e, True)
		rv2.vertex.add_link(rv1.vertex, e, False)
		return e

	def unlink(self):
		self.origin.remove_link(self.end)
		self.end.remove_link(self.origin)

		if self.hasPos():
			self.pos.delete()

		# Making unusable edge
		self.origin = self.end = None


	def hasPos():
		return self.pos is not None

	def common_verts(self):
		for n1 in self.origin.neighbours():
			for n2 in self.end.neighbours():
				if n1 == n2:
					yield n1


	def common_links(self):
		for n1 in self.origin.vertex.links:
			for n2 in self.end.vertex.links:
				if n1.other is n2.other:
					yield n1, n2

	def flip(self):
		other_verts = list(self.common_verts())

		if len(other_verts) != 2:
			raise Exception("There was more than 2 common verts!")

		firstV, secondV = other_verts

		e = Edge.link(firstV, secondV)
		if self.hasPos():
			e.pos = self.pos.append(e)

		self.unlink()

	def findRBRepr(self, pt, relToAbs):
		absToRel = np.linalg.inv(relToAbs)
		closestIntegerPt = absToRel.dot(pt - self.origin.coords) // 1 + 1
		print(closestIntegerPt)
		return self.origin.offset(closestIntegerPt), self.end.offset(closestIntegerPt)



class Link:

	def __init__(self, other, edge, isOrigin):
		self.other = other
		self.edge = edge
		self.isOrigin = isOrigin

	@property
	def offset(self):
		return self.edge.offset if self.isOrigin else - self.edge.offset

