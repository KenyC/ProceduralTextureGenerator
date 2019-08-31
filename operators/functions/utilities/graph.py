import numpy as np
from PIL import Image, ImageDraw
from itertools import chain
from list import DoubleList as DList, StableFirst 
from math import sqrt
from geometry import *
from periodic_component import *


class Graph:

	def __init__(self, vertices = [], edges = []):

		self.vertices = StableFirst()
		for vertex in vertices:
			self.register_vertex(vertex)

		self.edges = StableFirst()
		for edge in edges:
			self.register_edge(edge)

		

	def draw_img(self, **kwargs):

		parameters = {"size":(200,200), "center":None, "thick":1,  "colorLine":(255, 255, 255), "colorBG":(0, 0, 0)}
		parameters.update(kwargs)

		if parameters["center"] is None:
			center = np.array([0., 0.])
		else:
			center = parameters["center"]

		img = Image.new("RGB", parameters["size"], parameters["colorBG"])
		draw = ImageDraw.Draw(img)

		for edge in self.edges:
			draw.line([tuple((edge.origin.coords - center).tolist()), tuple((edge.end.coords - center).tolist())],
			 width = parameters["thick"], fill = parameters["colorLine"])

		return img

	def draw_forces(self, **kwargs):

		parameters = {"sizeForce" : 10., "center": None}
		parameters.update(kwargs)

		if parameters["center"] is None:
			center = np.array([0., 0.])
		else:
			center = parameters["center"]

		img = self.draw_img(**parameters)
		draw = ImageDraw.Draw(img)


		for vert in self.vertices:
			draw.line([tuple((vert.coords - center).tolist()),
							 tuple((vert.coords + vert.force * sizeForce - center).tolist())],
								 width = thick, fill = (0,255,0))

		return img

	def register_vertex(self, vertex):
		vertex.pos = self.vertices.append(vertex)

	def register_edge(self, edge):
		edge.pos = self.edges.append(edge)

	

	def summary(self, verbose =  False):

		print("Vertices {}", len(self.vertices))
		if verbose:
			for vertex, _ in zip(self.vertices, range(10)):
				print(vertex)
			print("...")

		print("Edges", len(self.edges))
		if verbose:
			for edge, _ in zip(self.edges, range(10)):
				print(edge)
			print("...")

		print("Areas", len(self.areas))
		if verbose:
			for area, _  in zip(self.areas, range(10)):
				print(area)
			print("...")

		print("Check vertices", all(vertex.pos is not None for vertex in self.vertices))
		print("Check edges", all(edge.pos is not None for edge in self.edges))
		print("Check areas", all(area.pos is not None for area in self.areas))



class PeriodicGraph(Graph):

	def __init__(self, vertices = [], edges = [], mat = None):
		super(PeriodicGraph, self).__init__(vertices, edges)
		self.toSq = mat
		self.periodX = mat[:,0]
		self.periodY = mat[:,1]

	def draw_img(self, size, center = None, thick = 1,  colorLine = (255, 255, 255), colorBG = (0, 0, 0)):
		if center is None:
			center = np.array([0., 0.])

		img = Image.new("RGB", size, colorBG)
		draw = ImageDraw.Draw(img)

		# Determine boundaries of image
		height, width = size
		eX = np.array([0, 1])
		eY = np.array([1, 0])
		boundaries = np.stack([center, center + width * eX, center + height * eY, center + width * eX + height * eY])


		origin, steps = findBoundingBox(boundaries, np.array([0., 0.]), self.toSq)

		stepX, stepY = tuple(int(x) for x in steps)
		print("origin")
		print(origin)
		print("step")
		print(stepX, stepY)

		for edge in self.edges:
			
			v1, v2 = edge.findRBRepr(origin, self.toSq)
		

	

			for i in range(stepX):
				for j in range(stepY):
					draw.line([tuple((v1.coords + i * self.periodX + j * self.periodY - center).tolist()),
							 tuple((v2.coords + i * self.periodX + j * self.periodY - center).tolist())],
								 width = thick, fill = (255,255,255))

		r = 3	
		for i in range(stepX):
			for j in range(stepY):
				draw.ellipse([tuple((origin + i * self.periodX + j * self.periodY - center - r).tolist()),
						 tuple((origin+ i * self.periodX + j * self.periodY - center + r).tolist())], fill = (255,0,0))
							

		return img

	def draw_non_periodic(self, *args, **kwargs):
		return super(PeriodicGraph, self).draw_img(*args, **kwargs)

	def draw_non_tiled(self, size, center = None, thick = 1,  colorLine = (255, 255, 255), colorBG = (0, 0, 0)):
		if center is None:
			center = np.array([0., 0.])

		img = Image.new("RGB", size, colorBG)
		imgs = []
		draw = ImageDraw.Draw(img)

		# Determine boundaries of image
		height, width = size
		eX = np.array([0, 1])
		eY = np.array([1, 0])
		boundaries = np.stack([center, center + width * eX, center + height * eY, center + width * eX + height * eY])


		origin = center

		for i, edge in enumerate(self.edges):
			v1 = findRBRepr(edge.vertex1.coords, origin, self.toSq)
			v2 = findClosestRepr(edge.vertex2.coords, v1, self.toSq)

			draw.line([tuple((v1  - center).tolist()),
						 tuple((v2 - center).tolist())],
						 width = thick, fill = (255,255,255))
		return img

	def draw_forces(self, size, center = None, sizeForce = 10., thick = 1,  colorLine = (255, 255, 255), colorBG = (0, 0, 0)):

		if center is None:
			center = np.array([0., 0.])

		img = self.draw_img(size, center, thick, colorLine, colorBG)
		draw = ImageDraw.Draw(img)

		# Determine boundaries of image
		height, width = size
		eX = np.array([0, 1])
		eY = np.array([1, 0])
		boundaries = np.stack([center, center + width * eX, center + height * eY, center + width * eX + height * eY])
		origin, steps = findBoundingBox(boundaries, np.array([0., 0.]), self.toSq)



		stepX, stepY = tuple(int(x) for x in steps)
		print("origin")
		print(origin)
		print("step")
		print(stepX, stepY)

		for vert in self.vertices:
			
			v1 = findRBRepr(vert.coords, origin, self.toSq)

			# if np.linalg.norm(v1 - v2) > self.l0:
			# 	print(np.linalg.norm(v1 - v2))
			# 	print(self.l0)
			# 	testV1, testV2, testSQ =  v1.copy(), edge.vertex2.coords.copy(), self.toSq.copy()
			# 	print("#CGREZ", v1, v2, edge.vertex2.coords)

			for i in range(stepX):
				for j in range(stepY):
					draw.line([tuple((v1 + i * self.periodX + j * self.periodY - center).tolist()),
							 tuple((v1 + vert.force * sizeForce + i * self.periodX + j * self.periodY - center).tolist())],
								 width = thick, fill = (0,255,0))

		return img
							


	def is_complete(self):
		return all(len(edge.areas) == 2 for edge in self.edges) and all(len(area.vertices) == 3 and len(area.edges) == 3 for area in self.areas)

	def rotateRandom(self, iterations = 5):
		l = len(self.edges) # The number of rotations stays constant through the process

		for i in range(iterations):
			rand = np.random.randint(l)
			self.edges.get(rand).rotate()

		assert(self.is_complete())
		
	def gradientDescent(self, l0 = 20., constant = 0.1, iter = 1, verbose = False):

		coords = np.stack([vertex.coords for vertex in self.vertices])
		forces = np.full_like(coords, 0., dtype = "float")


		for i, vertex in enumerate(self.vertices):
			for edge in vertex.edges:
				other = edge.vertex1 if vertex is edge.vertex2 else edge.vertex2 
				# Dist has to be evaluated taking into account periodicity
				unit = findClosestRepr(other.coords, vertex.coords, self.toSq) - vertex.coords
				dist = np.linalg.norm(unit)

				unit = unit / dist
				
				forces[i] = forces[i] + (dist - l0) * constant * unit
			if verbose:
				print("Vertex: ", vertex.coords," ==> force ", forces[i])

		coords = coords + forces

		for i, vertex in enumerate(self.vertices):
			vertex.coords = coords[i]
			vertex.force = forces[i]

		return np.linalg.norm(forces)


if __name__ == "__main__":

	scale = 50
	offset = 60

	mat = np.identity(2) * scale

	# vs = {"va" : Vertex(coords = np.array([0.37, 0.29], dtype = "float") * scale + offset).r(mat),
	# "vb" : Vertex(coords = np.array([0.77, 1.45], dtype = "float") * scale + offset).r(mat),
	# "vc" : Vertex(coords = np.array([2.1, 1.3], dtype = "float") * scale + offset).r(mat),
	# "vd" : Vertex(coords = np.array([2.1, 0.3], dtype = "float") * scale + offset).r(mat),
	# "ve" : Vertex(coords = np.array([1.4, 0.], dtype = "float") * scale + offset).r(mat),
	# "vf" : Vertex(coords = np.array([0.9, 0.1], dtype = "float") * scale + offset).r(mat),
	# "vg" : Vertex(coords = np.array([0.8, 0.65], dtype = "float") * scale + offset).r(mat),
	# "vh" : Vertex(coords = np.array([1.5, 1.1], dtype = "float") * scale + offset).r(mat),
	# "vi" : Vertex(coords = np.array([1.8, 1.1], dtype = "float") * scale + offset).r(mat)}

	# es = [	Edge.link(vs["va"], vs["vb"]),
	# Edge.link(vs["vb"], vs["vh"]),
	# Edge.link(vs["vh"], vs["vc"]),
	# Edge.link(vs["vc"], vs["vd"]),
	# Edge.link(vs["vd"], vs["ve"]),
	# Edge.link(vs["ve"], vs["vf"]),
	# Edge.link(vs["vf"], vs["va"]),
	# Edge.link(vs["vh"], vs["vi"]),
	# Edge.link(vs["vi"], vs["vc"]),
	# Edge.link(vs["vi"], vs["vd"]),
	# Edge.link(vs["vi"], vs["vg"]),
	# Edge.link(vs["vg"], vs["vh"]),
	# Edge.link(vs["vg"], vs["ve"]),
	# Edge.link(vs["vg"], vs["vf"]),
	# Edge.link(vs["va"], vs["vg"]),
	# Edge.link(vs["vg"], vs["vb"]),
	# Edge.link(vs["vd"], vs["vg"])]

	# graph1 = Graph(vs.values(), es)
	# img1 = graph1.draw_img(size = (200,200))

	# Second graph
	ex = np.array([1., 0.])
	ey = - np.array([0., 1.])

	vs = {"va" : Vertex(coords = np.array([0.05, 1.04], dtype = "float") * scale + offset).r(mat),
	"vb" : Vertex(coords = np.array([0.47, 0.88], dtype = "float") * scale + offset).r(mat),
	"vc" : None,
	"vd" : Vertex(coords = np.array([0.17, 0.56], dtype = "float") * scale + offset).r(mat),
	"ve" : Vertex(coords = np.array([0.38, 0.45], dtype = "float") * scale + offset).r(mat),
	"vf" : Vertex(coords = np.array([0.73, 0.66], dtype = "float") * scale + offset).r(mat),
	"vg" : Vertex(coords = np.array([0.86, 0.35], dtype = "float") * scale + offset).r(mat)}

	vs["vc"] = vs["va"].offset(ex)


	es = [	Edge.link(vs["va"], vs["vb"]),
	Edge.link(vs["vb"], vs["vc"]),
	Edge.link(vs["vc"], vs["vd"].offset(ex)),
	Edge.link(vs["vd"], vs["va"].offset(ey)),
	Edge.link(vs["ve"], vs["va"].offset(ey)),
	Edge.link(vs["ve"], vs["vg"]),
	Edge.link(vs["ve"], vs["vf"]),
	Edge.link(vs["ve"], vs["vb"]),
	Edge.link(vs["ve"], vs["vb"].offset(ey)),
	Edge.link(vs["ve"], vs["vd"]),
	Edge.link(vs["vg"], vs["vf"]),
	Edge.link(vs["vg"], vs["vd"].offset(ex)),
	Edge.link(vs["vg"], vs["vc"].offset(ey)),
	Edge.link(vs["vg"], vs["vb"].offset(ey)),
	Edge.link(vs["vb"], vs["vd"]),
	Edge.link(vs["vf"], vs["vd"].offset(ex)),
	Edge.link(vs["vf"], vs["vc"]),
	Edge.link(vs["vf"], vs["vb"])]

	graph1 = PeriodicGraph(vs.values(), es, mat)
	img1 = graph1.draw_img(size = (200,200))

	