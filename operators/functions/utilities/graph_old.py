import numpy as np
from PIL import Image, ImageDraw
from itertools import chain
from list import DoubleList as DList, StableFirst 
from math import sqrt
from geometry import *



class EdgeNotFound(Exception):
	pass

class Vertex:

	def __init__(self, coords, edges = [], pos = None):
		self.edges = edges if edges else []
		self.coords = coords
		self.pos = pos

	def add_edge(self, edge):
		self.edges.append(edge)

	def remove_edge(self, toRemove):
		for i, edge in enumerate(self.edges):
			if edge is toRemove:
				del self.edges[i]
				return

	# Returns the edge between self and other
	def find_edge(self, other):
		for edge in self.edges:
			if edge.vertex1 is other or edge.vertex2 is other:
				return edge

		raise EdgeNotFound

	def merge(v1, v2):

		v3 = Vertex(v1.coords)

		areas_to_collapse = []
		for edge in chain(iter(v1.edges), iter(v2.edges)):

		

			

			b = True
			for i, vertex in enumerate(edge.vertices):
				if vertex is v1 or vertex is v2:
					edge.vertices[i] = v3
				else:
					b = False


			#Hairy case: if there is an edge between the vertices to be merged
			if b:
				if edge.pos is not None:
					edge.pos.delete()
					# print("Deleted an edge")
				areas_to_collapse += edge.areas
			else:
				v3.add_edge(edge)


			for area in edge.areas:
				for i, vertex in enumerate(area.vertices):
					if vertex is v1 or vertex is v2:
						area.vertices[i] = v3


		for area in areas_to_collapse:
			
			# Find edge (v3, v3) as well as other edges
			otherEdges = []
			for idx1, edge1 in enumerate(area.edges):
				if edge1.vertex1 is not v3:
					otherEdges.append(edge1)
					otherV = edge1.vertex1
				elif edge1.vertex2 is not v3:
					otherEdges.append(edge1)
					otherV = edge1.vertex2
				else:
					edge = edge1
					idx = idx1	

			edgeCollapse = Edge.make_edge(v3, otherV)
			hasBeenAppended = False

			# Find neighbouring areas and replace the other edges with the new collapsed edge
			for edge in otherEdges:
				print("Looping over that", edge)
				for area1 in edge.areas:
					for idx, edge1 in enumerate(area.edges):
						if edge1 is edge and area is not area1:
							area.edges[idx] = edgeCollapse
				if edge.pos is not None:
					if not hasBeenAppended:
						edge.pos.append(edgeCollapse)
						hasBeenAppended = True
					edge.pos.delete()
					# print("Deleted an edge: ", edge)

			if area.pos is not None:
				# print("Deleted an area")
				area.pos.delete()

		# print("###",v1.pos,v2.pos)
		if v1.pos is not None and v2.pos is not None:
			v3.pos = v1.pos.append(v3)
			v1.pos.delete()
			v2.pos.delete()
			# print("BHDT11")
		elif v1.pos is not None:
			v1.pos.append(v3)
			v1.pos.delete()
		elif v2.pos is not None:
			v2.pos.append(v3)
			v2.pos.delete()

		return v3


	def __str__(self):
		return "({},{})".format(self.coords[0], self.coords[1])



class Edge:

	def __init__(self, vertices = [], areas = [], pos = None):
		self.vertices = vertices if vertices else []
		self.areas = areas if areas else []
		self.pos = pos

	@property
	def length(self):
		return np.linalg.norm(self.vertex1.coords - self.vertex2.coords)
	

	@property
	def vertex1(self):
		return self.vertices[0]

	@vertex1.setter
	def vertex1(self, value):
		self.vertices[0] = value
	
	@property
	def vertex2(self):
		return self.vertices[1]

	@vertex2.setter
	def vertex2(self, value):
		self.vertices[1] = value

	def add_area(self, area):
		if len(self.areas) < 2:
			self.areas.append(area)
		else:
			raise Exception("This edge has already two areas")
		

	def add_vertex(self, vertex):
		if len(self.vertices) < 2:
			self.vertices.append(vertex)
		else:
			raise Exception("This edge has already two vertices")

	def make_edge(vertex1, vertex2):
		edge = Edge([vertex1, vertex2])
		vertex1.add_edge(edge)
		vertex2.add_edge(edge)

		return edge

	def remove_area(self, toRemove):
		for i, area in enumerate(self.areas):
			if area is toRemove:
				del self.areas[i]
				return

	"""
	In a complete triangulation, edges border two tringular areas. They are the diagonal of a qudrilater
	Rotate replaces an edge with the other diagonal
	"""
	def rotate(self):
		# Checksum : 2 areas removed, 2 areas added, 1 edge removed, 1 edge added
		print(self)
		# Find the other vertices
		firstV, secondV = self.findFlipVerts()

		try:
			firstV.find_edge(secondV)
		except EdgeNotFound:
			pass
		else:
			print("Couldn't rotate this one.")
			return

		# Remove oneself from the vertices
		for vert in self.vertices:
			vert.remove_edge(self)

		# Remove the areas from adjacent edges
		for area in self.areas:
			for edge in area.edges:
				if edge is not self: # We need a reference to the areas for later
					edge.remove_area(area)


		# Make an edge between the other verts and add it to the double list
		nEdge = Edge.make_edge(firstV, secondV)
		nEdge.pos = self.pos.append(nEdge)

		# Make two new areas and add them to the double list
		nArea1 = Area.make_area(firstV, secondV, self.vertex1)
		nArea1.pos = self.areas[0].pos.append(nArea1)
		nArea2 = Area.make_area(firstV, secondV, self.vertex2)
		nArea2.pos = self.areas[0].pos.append(nArea2)

		# Remove edge, and original two areas from double list
		for area in self.areas:
			area.delete()
		self.delete()

		return nEdge


	def delete(self):
		self.pos.delete()
		self.vertices = []
		self.areas = []



	def findFlipVerts(self):
		otherVerts = []
		for area in self.areas:
			for vert in area.vertices:
				if not self.contains(vert):
					otherVerts.append(vert)

		# if len(otherVerts) > 2:
		# 	raise Exception("Graph was abnormal ; too many edges or too many verts")
		# else:
		# 	firstV, secondV = otherVerts
		return tuple(otherVerts)
		

	def contains(self, vertex):
		return (vertex is self.vertex1) or (vertex is self.vertex2)

	def __str__(self):
		return "{} -> {}".format(str(self.vertex1), str(self.vertex2))

class Area:

	def __init__(self, vertices = [], edges = [], pos = None):
		self.vertices = vertices if vertices else []
		self.edges = edges if edges else []
		self.pos = pos


	def add_edge(self, edge):
		if len(self.edges) < 3:
			self.edges.append(edge)
		else:
			raise Exception("This area has already three edges")
		

	def add_vertex(self, vertex):
		if len(self.vertices) < 3:
			self.vertices.append(vertex)
		else:
			raise Exception("This area has already three vertices")

	def delete(self):
		self.pos.delete()
		self.vertices = []
		self.edges = []

	def make_area(v1, v2, v3):
		area = Area(vertices = [v1, v2, v3])

		edges = [v1.find_edge(v2), v1.find_edge(v3), v2.find_edge(v3)]

		for edge in edges:
			area.add_edge(edge)
			edge.add_area(area)


		return area

	def replace_edge(self, object, replacement):
		for i, edge in self.edges:
			if edge is object:
				self.edges[i] = replacement
				return

	def __str__(self):
		return " - ".join([str(vert) for vert in self.vertices])


class Graph:

	def __init__(self, vertices = [], edges = [], areas = []):

		self.vertices = StableFirst()
		for vertex in vertices:
			self.register_vertex(vertex)

		self.edges = StableFirst()
		for edge in edges:
			self.register_edge(edge)

		self.areas = StableFirst()
		for area in areas:
			self.register_area(area)
		

	def draw_img(self, size , center = None, thick = 1,  colorLine = (255, 255, 255), colorBG = (0, 0, 0)):
		if center is None:
			center = np.array([0., 0.])

		img = Image.new("RGB", size, colorBG)
		draw = ImageDraw.Draw(img)

		for edge in self.edges:
			draw.line([tuple((edge.vertex1.coords - center).tolist()), tuple((edge.vertex2.coords - center).tolist())], width = thick, fill = (255,255,255))

		return img

	def draw_forces(self, size, center = None, sizeForce = 10., thick = 1,  colorLine = (255, 255, 255), colorBG = (0, 0, 0)):

		if center is None:
			center = np.array([0., 0.])

		img = self.draw_img(size, center, thick, colorLine, colorBG)
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


	def register_area(self, area):
		area.pos = self.areas.append(area)

	def identify_vertex(self, pos1, pos2):
		newV, newEdges = Vertex.merge(pos1.value, pos2.value)

	def gradientDescent(self, l0 = 20., constant = 0.1, iter = 1, verbose = True):
		

		coords = np.stack([vertex.coords for vertex in self.vertices])
		forces = np.full_like(coords, 0., dtype = "float")

		for i, vertex in enumerate(self.vertices):
			for edge in vertex.edges:
				other = edge.vertex1 if vertex is edge.vertex2 else edge.vertex2 
				unit = other.coords - vertex.coords

				dist = np.linalg.norm(unit)
				unit = unit / dist
				
				forces[i] = forces[i] + (dist - l0) * constant * unit
			if verbose:
				print("Vertex: ", vertex.coords," ==> force ", forces[i])

		coords = coords + forces

		for i, vertex in enumerate(self.vertices):
			vertex.coords = coords[i]




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

	def __init__(self):
		raise Exception("Underivable class")

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
			
			v1 = findRBRepr(edge.vertex1.coords, origin, self.toSq)
			v2 = findClosestRepr(edge.vertex2.coords, v1, self.toSq)

			if abs(np.linalg.norm(v2 - v1) - np.linalg.norm(v1 - findRBRepr(edge.vertex2.coords, origin, self.toSq))) < 1e-5:
				print("Did it")
				v2 = findRBRepr(edge.vertex2.coords, origin, self.toSq)

			# if np.linalg.norm(v1 - v2) > self.l0:
			# 	print(np.linalg.norm(v1 - v2))
			# 	print(self.l0)
			# 	testV1, testV2, testSQ =  v1.copy(), edge.vertex2.coords.copy(), self.toSq.copy()
			# 	print("#CGREZ", v1, v2, edge.vertex2.coords)

			for i in range(stepX):
				for j in range(stepY):
					draw.line([tuple((v1 + i * self.periodX + j * self.periodY - center).tolist()),
							 tuple((v2+ i * self.periodX + j * self.periodY - center).tolist())],
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

class ToricTriangularGraph(PeriodicGraph):

	DIRECTIONS = [np.array([1.0, 0.0]), np.array([0.5, sqrt(3)/2]), np.array([-0.5, sqrt(3/2)]),
				 np.array([-1.0, 0.0]), np.array([-0.5, -sqrt(3)/2]), np.array([0.5, -sqrt(3/2)])]


	X_DIR = DIRECTIONS[0]
	Y_DIR = DIRECTIONS[1]
	# size is number of vertices on a side
	def __init__(self, size,  l0 = 20.):
		super(PeriodicGraph, self).__init__()
		# Create non toric graph
		self.verts = [[Vertex(coords = i * ToricTriangularGraph.X_DIR + j * ToricTriangularGraph.Y_DIR) for i in range(size)] for j in range(size)]

		self.l0 = l0

		for i in range(size):
			for j in range(size):

				vertex = self.verts[i][j]
				self.register_vertex(vertex)

				# from this vertex we construct the edges of the triangle of which this vertex is the bottom left
				vUp = self.verts[(i+1) % size][j]
				vRight = self.verts[i][(j+1) % size]
				self.register_edge(Edge.make_edge(vertex, vUp))
				self.register_edge(Edge.make_edge(vertex, vRight))
				self.register_edge(Edge.make_edge(vUp, vRight))


				# From this vertex, we construct the triangle of which it is the bottom left  corner
				vDown = self.verts[(i+1) % size][(j-1) % size]
				self.register_area(Area.make_area(vertex, vUp, vRight))

		# When all edges are complete, we create the triangles that point down,
		# each vertex constructs the triangle of which it is the upper left corner
		for i in range(size):
			for j in range(size):
				vertex = self.verts[i][j]
				vDown = self.verts[(i-1) % size][(j+1) % size]
				vRight = self.verts[i][(j+1) % size]
				self.register_area(Area.make_area(vertex, vDown, vRight))

		for vert in self.vertices:
			vert.coords = vert.coords * l0


		self.summary()
		assert(self.is_complete())

		self.mutable_edges = self.edges
		self.periodX  = size  * l0 * ToricTriangularGraph.X_DIR
		self.periodY  = size  * l0 * ToricTriangularGraph.Y_DIR
		self.toSq = np.array([self.periodX, self.periodY]).T 



	






if __name__ == "__main__":
	v1 = Vertex(coords = np.array([0., 1.], dtype = "float") * 50 + 60)
	v2 = Vertex(coords = np.array([1., 0.], dtype = "float") * 50 + 60)
	v3 = Vertex(coords = np.array([1., 1.], dtype = "float") * 50 + 60)
	v4 = Vertex(coords = np.array([0., 0.], dtype = "float") * 50 + 60)

	e1 = Edge.make_edge(v1, v2)
	e2 = Edge.make_edge(v2, v3)
	e3 = Edge.make_edge(v3, v1)
	e4 = Edge.make_edge(v4, v1)
	e5 = Edge.make_edge(v4, v2)

	a1 = Area.make_area(v1, v2, v3)
	a2 = Area.make_area(v1, v2, v4)

	vertices = DList.fromList([v1, v2, v3, v4])
	edges = DList.fromList([e1, e2, e3, e4, e5])
	areas = DList.fromList([a1, a2])

	graph1 = Graph(vertices, edges, areas)
	img1 = graph1.draw_img((200,200))

	#print("Merging {} and {}".format(v1, v3))
	#Vertex.merge(v1, v3)
	img2 = graph1.draw_img((200,200))	

	graph2 = ToricTriangularGraph(5, l0 = 10)
	img3 = graph2.draw_img((200,200))

