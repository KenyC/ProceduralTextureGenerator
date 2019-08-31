from graph import Graph, Vertex, Edge, Area
import numpy as np

class Test(Graph):

		def __init__(self, *args, **kwargs):
			super(Test, self).__init__(*args, **kwargs)

		def gradientDescent(self, dynamic, speed = 0.1, verbose = False):

			self.initForces()

			if dynamic.perWhat == "point":
				for vertex in self.vertices:
					dynamic(vertex)
			elif dynamic.perWhat == "edge":
				for edge in self.edges:
					dynamic(edge)
			elif dynamic.perWhat == "area":
				for area in self.areas:
					dynamic(area)

			forces = np.full((len(self.vertices), 2), 0.0)
			for i, vertex in enumerate(self.vertices):
				vertex.coords = vertex.coords + speed * vertex.force
				forces[i] = vertex.force

			return np.linalg.norm(forces)

		def initForces(self):

			for vertex in self.vertices:
				vertex.force = np.zeros(2)

class Dynamic:

	def __init__(self, perWhat):
		self.perWhat = perWhat

	def force(self, element):
		return

	def __call__(self, element):
		self.force(element)

class SpringDynamic(Dynamic):
	"""	"""
	def __init__(self, l0 = 0.):
		super(SpringDynamic, self).__init__("edge")
		self.l0 = l0
	
	def force(self, edge):
		dist = edge.length
		unit = (edge.vertex2.coords - edge.vertex1.coords) / dist

		edge.vertex1.force += (dist - self.l0) * unit
		edge.vertex2.force += - (dist - self.l0) * unit


class RepelEdges(Dynamic):

	ROT_MATRIX = np.array([[0, 1],[-1, 0]], dtype = "float")
	"""	"""
	def __init__(self):
		super(RepelEdges, self).__init__("edge")
		
	def force(self, edge):
		
		unit = (edge.vertex2.coords - edge.vertex1.coords) / edge.length
		flipVerts = edge.findFlipVerts()

		for vert in flipVerts:
			pass
			# direction = RepelEdges.ROT_MATRIX.dot()
			# np.cross(unit)


		
if __name__ == "__main__":

	scale = 50
	offset = 60

	vs = {"va" : Vertex(coords = np.array([0.37, 0.29], dtype = "float") * scale + offset),
	"vb" : Vertex(coords = np.array([0.77, 1.45], dtype = "float") * scale + offset),
	"vc" : Vertex(coords = np.array([2.1, 1.3], dtype = "float") * scale + offset),
	"vd" : Vertex(coords = np.array([2.1, 0.3], dtype = "float") * scale + offset),
	"ve" : Vertex(coords = np.array([1.4, 0.], dtype = "float") * scale + offset),
	"vf" : Vertex(coords = np.array([0.9, 0.1], dtype = "float") * scale + offset),
	"vg" : Vertex(coords = np.array([0.8, 0.65], dtype = "float") * scale + offset),
	"vh" : Vertex(coords = np.array([1.5, 1.1], dtype = "float") * scale + offset),
	"vi" : Vertex(coords = np.array([1.8, 1.1], dtype = "float") * scale + offset)}

	es = [	Edge.make_edge(vs["va"], vs["vb"]),
	Edge.make_edge(vs["vb"], vs["vh"]),
	Edge.make_edge(vs["vh"], vs["vc"]),
	Edge.make_edge(vs["vc"], vs["vd"]),
	Edge.make_edge(vs["vd"], vs["ve"]),
	Edge.make_edge(vs["ve"], vs["vf"]),
	Edge.make_edge(vs["vf"], vs["va"]),
	Edge.make_edge(vs["vh"], vs["vi"]),
	Edge.make_edge(vs["vi"], vs["vc"]),
	Edge.make_edge(vs["vi"], vs["vd"]),
	Edge.make_edge(vs["vi"], vs["vg"]),
	Edge.make_edge(vs["vg"], vs["vh"]),
	Edge.make_edge(vs["vg"], vs["ve"]),
	Edge.make_edge(vs["vg"], vs["vf"]),
	Edge.make_edge(vs["va"], vs["vg"]),
	Edge.make_edge(vs["vg"], vs["vb"]),
	Edge.make_edge(vs["vd"], vs["vg"])]
	

	ars = [	Area.make_area(vs["va"], vs["vb"], vs["vg"]),
	Area.make_area(vs["vb"], vs["vg"], vs["vh"]),
	Area.make_area(vs["vg"], vs["vh"], vs["vi"]),
	Area.make_area(vs["vh"], vs["vi"], vs["vc"]),
	Area.make_area(vs["vi"], vs["vc"], vs["vd"]),
	Area.make_area(vs["vi"], vs["vd"], vs["vg"]),
	Area.make_area(vs["ve"], vs["vd"], vs["vg"]),
	Area.make_area(vs["vg"], vs["ve"], vs["vf"]),
	Area.make_area(vs["va"], vs["vg"], vs["vf"])]

	test = Test(vs.values(), es, ars)