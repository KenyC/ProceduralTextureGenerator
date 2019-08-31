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