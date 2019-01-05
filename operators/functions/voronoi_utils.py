from queue import PriorityQueue
import numpy as np

class MultipleTargets(Exception):

	def __init__(self, tree):
		self.tree = tree


# Returns x of intersection of horizontal line y with parabola
def parabolaPt(locus, x_directrix, y):
	# Point (x1,y1) on parabola 1 such that y1 = y 
	return (locus.x + x_directrix) / 2 + (locus.y - y) **2 / (2 * (locus.x - x_directrix))


def dist(pt1, pt2):
	return ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5



# Returns True if y is just below parabola 1
def parabolaIntersect(locus1, locus2, x_directrix, y):

	# Point (x1,y1) on parabola 1 such that y1 = y 
	x1 = parabolaPt(locus1, x_directrix, y)

	# Point (x2,y2) on parabola 2 such that y2 = y 
	x2 = parabolaPt(locus2, x_directrix, y)

	if x1 == x2:
		raise LookupError

	return (x1 == max(x1, x2))


def circum_circle(pt1, pt2, pt3):
	# Formula for Cartesian coordinates of center from Wikipedia
	D = 2 * (pt1.x * (pt2.y - pt3.y) + pt2.x * (pt3.y - pt1.y) + pt3.x * (pt1.y - pt2.y))

	a = pt1.x**2 + pt1.y**2
	b = pt2.x**2 + pt2.y**2
	c = pt3.x**2 + pt3.y**2

	try:
		x = (1/D) * (a * (pt2.y - pt3.y) + b * (pt3.y - pt1.y) + c * (pt1.y - pt2.y))
		y = -(1/D) * (a * (pt2.x - pt3.x) + b * (pt3.x - pt1.x) + c * (pt1.x - pt2.x))
	except ZeroDivisionError:
		x = (pt1.x + pt2.x + pt3.x)/3
		y = (pt1.y + pt2.y + pt3.y)/3
		print("Points ({},{}), ({},{}) and ({},{}) are colinear".format(pt1.x, pt1.y, pt2.x, pt2.y, pt3.x, pt3.y))

	return x, y, ((pt1.x - x)**2 + (pt1.y - y)**2)**0.5


class Queue(PriorityQueue):

	def __init__(self, *args):
		super(Queue, self).__init__(*args)

	def __iter__(self):
		return self

	def __next__(self):
		if self.empty():
			raise StopIteration
		else:
			return self.get()

class Pt:

	def __init__(self, idx, voronoi):
		self.v = voronoi
		self.idx = idx

	@property
	def x(self):
		return self.v.pts[self.idx][0]

	@property
	def y(self):
		return self.v.pts[self.idx][1]

	def __getitem__(self, idx):
		if idx == 0:
			return self.x
		elif idx == 1:
			return self.y
		else:
			raise LookupError

	def __eq__(self, other):
		return self.idx == other.idx

	def __str__(self):
		return str(self.idx)


class NoIntersectionBetweenRays(Exception):
	pass


class Edge:

	# None represents a point at infinity
	def __init__(self, pt1, pt2, origin, direction):
		self.pt1, self.pt2 = pt1, pt2
		self.origin = origin
		self.dir = direction
		self.boundary = None

	def add_boundary(self, bound):
		self.boundary = bound

	def is_complete(self):
		return not (self.boundary is None)

	def __str__(self):
		return "{} -> {} (orig = {}, dir = {})".format(self.pt1, self.pt2, self.origin, self.dir)

	def intersect(self, other):
		matrix = np.matrix([[self.dir[0], -other.dir[0]],[self.dir[1], -other.dir[1]]])
		origSelf = np.array(self.origin).reshape((2,))
		origOther = np.array(other.origin).reshape((2,))
		
		if np.linalg.det(matrix) == 0.:
			raise NoIntersectionBetweenRays
		else:

			points = np.linalg.inv(matrix). dot(origOther - origSelf)
			
			if any(x<0 for x in points.flat):
				raise NoIntersectionBetweenRays
			else:
				print(matrix.T[0])
				print(origSelf)
				print(points[0, 0])
				return tuple((origSelf + points[0, 0] * matrix.T[0]).flat)

	def opposite(self):
		return Edge(self.pt1, self.pt2, self.origin, (-self.dir[0], -self.dir[1]))











class SearchTree:

	def __init__(self, parent = None, focus = None, children = []):

		self.focus = focus
		
		
		
		self.parent = parent
		self.edge = None

		self.children = children
		



	def __str__(self):


		def str_rec(tree, n):
			s = n*"\t" + "[parent: {}, focus: {}, edge: {}]\n".format(tree.parent is not None,
															 tree.focus,
															 tree.edge)
			for child in tree.children:
				s += str_rec(child, n+1)

			return s

		return str_rec(self, 0)


	@property
	def children(self):
		return self._children
		
	@children.setter
	def children(self, value):
		self._children = value
		for child in self._children:
			child.parent = self

	@property
	def sister(self):
		if self.parent is None:
			raise LookupError
		else:
			return self.parent.leftChild if self is self.parent.rightChild else self.parent.rightChild
	

	@property
	def leftChild(self):
		return self.children[0]

	@leftChild.setter
	def leftChild(self, value):
		self.children[0] = value
		value.parent = self
	
	
	@property
	def rightChild(self):
		return self.children[1]

	@rightChild.setter
	def rightChild(self, value):
		self.children[1] = value
		value.parent = self
	
	@property
	def leftOf(self):
		return self.leftSCA.leftChild.rightmost
	
	@property
	def rightOf(self):
		return self.rightSCA.rightChild.leftmost

	@property
	def leftSCA(self):
		if self.parent is None:
			raise LookupError
		if self.is_right_child():
			return self.parent
		else:
			return self.parent.leftSCA

	
	@property
	def rightSCA(self):
		if self.parent is None:
			raise LookupError
		if not self.is_right_child():
			return self.parent
		else:
			return self.parent.rightSCA

	@property
	def leftrightSCA(self):
		return self.rightSCA if self.is_right_child() else self.leftSCA

	@property
	def leftmost(self):
		if self.is_leaf():
			return self
		else:
			return self.leftChild.leftmost
	
	@property
	def rightmost(self):
		if self.is_leaf():
			return self
		else:
			return self.rightChild.rightmost

	@property
	def root(self):
		if self.parent is None:
			return self
		else:
			return self.parent.root
	def is_leaf(self):
		return not self.children

	def is_right_child(self):
		if self.parent is None:
			raise LookupError
		else:
			return self.parent.rightChild is self


	
	def find(self, x_ligne, y_pt):
		if self.is_leaf():
			return self, parabolaPt(self.focus, x_ligne, y_pt)
		else:
			leftN = self.leftChild.rightmost
			rightN = self.rightChild.leftmost

			# Find intersection of parabolas (directrix = x_ligne)
			
			try:
				belowLeft = parabolaIntersect(leftN.focus, rightN.focus, x_ligne, y_pt)
			except LookupError:
				belowLeft = False
					
			if belowLeft:
				return self.leftChild.find(x_ligne, y_pt)
			else:
				return self.rightChild.find(x_ligne, y_pt)
			


class VoronoiGraph:

	def __init__(self, pts):
		self.pts = pts
		# Sort by x
		self.pts.sort(key = lambda x: x[0])

		self.edges = []
		self.boundaries = []

		self.compute_graph()

	def compute_graph(self):
		queue = Queue()
		tree = SearchTree(focus = Pt(0, self))

		def check_circle(node, xCurrent):
			try:
				left = node.leftSCA
				right = node.rightSCA
			except LookupError:
				return
			else:
				print("Checking {} and {}".format(left.edge, right.edge))
				try:
					pt = x, y = left.edge.intersect(right.edge)
				except NoIntersectionBetweenRays:
					return
				else:						
					r = dist(node.focus, pt)
					if x+r > xCurrent:
						print("Points", node.focus, left.focus, right.focus)
						print(x,y,r)
						queue.put(Delete(x+r, node))
				

		class Event:
			def __init__(self):
				raise Exception("Can't derive base class")
			def __lt__(self, other):
				return self.x < other.x

			def __gt__(self, other):
				return self.x > other.x

			def __leq__(self, other):
				return self.x <= other.x

			def __geq__(self, other):
				return self.x >= other.x


		class Delete(Event):

			def __init__(self, x, treeNode):
				self.x = x
				self.treeNode = treeNode

			def execute(selfEv):

				print("DELETE ", selfEv.treeNode.leftOf.focus, selfEv.treeNode.focus, selfEv.treeNode.rightOf.focus)
				
				
				# For adding boundary node to outcoming edge
				right = selfEv.treeNode.rightOf
				left = selfEv.treeNode.leftOf
				focusLeft = left.focus
				focusRight = right.focus
				sca = selfEv.treeNode.leftrightSCA


				# Adding boundaries to collapsing edges
				try:
					edge1 = selfEv.treeNode.leftSCA.edge
					edge2 = selfEv.treeNode.rightSCA.edge
					
					intersect = edge1.intersect(edge2)

					edge1.add_boundary(intersect)
					edge2.add_boundary(intersect)
					self.boundaries.append(intersect)
				except AttributeError:
					print(tree)
					raise AttributeError
				except NoIntersectionBetweenRays:
					raise Exception("why was DELETE ever called?")

				# "treeNode"'s parent must be replaced with "treeNode"'s sister
				parentReplace = selfEv.treeNode.parent.parent
				daughterReplace = selfEv.treeNode.sister

				if selfEv.treeNode.parent.is_right_child():
					parentReplace.rightChild = daughterReplace
				else:
					parentReplace.leftChild = daughterReplace

				origin = intersect
				# Direction should be toward x positive
				dir1 = -(focusLeft.y - focusRight.y), (focusLeft.x - focusRight.x)
				if dir1[0] < 0:
					dir1 = -dir1[0], -dir1[1]
				
				edge = Edge(focusLeft, focusRight, origin, dir1)
				# This edge starts at a boundary
				edge.add_boundary(origin)
				self.edges.append(edge)
				sca.edge = edge

				# Add edge

				print(tree)

				check_circle(parentReplace.rightChild.leftmost, selfEv.x)
				check_circle(parentReplace.leftChild.rightmost, selfEv.x)
				

		class Insert(Event):
			def __init__(self, focus):
				self.tree = tree

				# self.focus is instance Pt
				self.focus = focus
				self.x = self.focus.x

			def execute(selfEv):
				
				try:
					node, xIntersect = tree.find(selfEv.x, selfEv.focus.y)
				except MultipleTargets as t:
					raise Exception("Multiple target found ; not implemented yet")
				else:
					print("INSERT", selfEv.focus.idx,"in", node.focus)
					
					# creating new right child of "node"
					st = SearchTree(children = [SearchTree(focus = selfEv.focus), SearchTree(focus = node.focus)])

					# Sprouting new children
					node.children = [SearchTree(focus = node.focus), st]

					# Inner nodes are labelled with an edge
					origin = xIntersect, selfEv.focus.y
					# Direction orthogonal to segment between foci
					dir1 = -(node.focus.y - selfEv.focus.y), (node.focus.x - selfEv.focus.x)
					# One edge is toward y positive
					if dir1[1] > 0:
						dir1 = -dir1[0], -dir1[1]

					edge = Edge(node.focus, selfEv.focus, origin, dir1)
					node.edge = edge

					opEdge = edge.opposite()
					st.edge = opEdge

					self.edges.append(opEdge)
					self.edges.append(edge)
					
					# Delete focus for inner node
					node.focus = None

					# Check for circle intersection
					check_circle(st.rightChild, selfEv.x)
					check_circle(node.leftChild, selfEv.x)

					print("Output insert:")
					print(tree)

				

		for i in range(len(self.pts) - 1, 0, -1):
			queue.put(Insert(Pt(i, self)))

		for event in queue:
			event.execute()
				
a = Edge(0,1,(0,0),(1,1))
b = Edge(0,1,(0,1),(2,-1))
#a.intersect(b)

testPts = [(-3., 1.), (-2., -1.), (3., 1.), (4., -1.)]
voronoi = VoronoiGraph(testPts)

print(voronoi.boundaries)
for edge in voronoi.edges:
	if edge.is_complete():
		print(edge)
	else:
		print("INCOMPLETE EDGE", edge)


