from graph import PeriodicGraph, Vertex, Edge, Area
from PIL import Image, ImageDraw
from geometry import *
import numpy as np

class  SquareGraph(PeriodicGraph):
	"""docstring for  SquareGraph"""
	DIRECTIONS = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([-1.0, 0.0]),
				 np.array([0.0, -1.0])]
	X_DIR = DIRECTIONS[0]
	Y_DIR = DIRECTIONS[1]

	def __init__(self, size,  l0 = 20.):
		super(PeriodicGraph, self).__init__()
		self.mutable_edges = []


		self.verts = [[Vertex(coords = i * SquareGraph.X_DIR + j * SquareGraph.Y_DIR) for i in range(size)] for j in range(size)]

		self.l0 = l0

		for i in range(size):
			for j in range(size):

				vertex = self.verts[i][j]
				self.register_vertex(vertex)

				# from this vertex we construct the edges of the triangle of which this vertex is the bottom left
				vUp = self.verts[(i+1) % size][j]
				vRight = self.verts[i][(j+1) % size]

				# edges that are not on the side are immutable
				edge = Edge.make_edge(vertex, vUp)
				self.register_edge(edge)
				if not(j == 0):
					self.mutable_edges.append(edge)
				
				edge = Edge.make_edge(vertex, vRight)
				self.register_edge(edge)
				if not(i == 0):
					self.mutable_edges.append(edge)
				
				edge = Edge.make_edge(vUp, vRight)
				self.register_edge(edge)
				self.mutable_edges.append(edge)

				# From this vertex, we construct the triangle of which it is the bottom left  corner
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

		self.periodX  = size  * l0 * SquareGraph.X_DIR
		self.periodY  = size  * l0 * SquareGraph.Y_DIR
		self.toSq = np.array([self.periodX, self.periodY]).T 

	def rotateRandom(self, iterations = 5):
		l = len(self.mutable_edges) # The number of rotations stays constant through the process

		for i in range(iterations):
			rand = np.random.randint(l)
			edge = self.mutable_edges[rand] 

			firstV, secondV = edge.findFlipVerts()

			path = edge.vertex1.coords, firstV.coords, edge.vertex2.coords, secondV.coords

			if is_convex(path):
				self.mutable_edges.append(edge.rotate())
				del self.mutable_edges[rand]

			else:
				print("Wasn't convex ; did not flip.")

		assert(self.is_complete())

	def draw_mutable(self, size, center = None, thick = 1,  colorLine = (255, 255, 255), colorBG = (0, 0, 0)):
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

		for edge in self.mutable_edges:
			v1 = findRBRepr(edge.vertex1.coords, origin, self.toSq)
			v2 = findClosestRepr(edge.vertex2.coords, v1, self.toSq)

			draw.line([tuple((v1  - center).tolist()),
						 tuple((v2 - center).tolist())],
						 width = thick, fill = (255,255,255))
		return img

if __name__ == "__main__":

	sqGraph = SquareGraph(size = 5, l0 = 30.)
