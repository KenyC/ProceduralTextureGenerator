from queue import PriorityQueue
from utilities.list import DoubleList
import numpy as np
from functools import reduce
from PIL import Image, ImageDraw

class MultipleTargets(Exception):

	def __init__(self, tree):
		self.tree = tree


# Returns x of intersection of horizontal line y with parabola
def parabolaPt(locus, x_directrix, y):
	# Point (x1,y1) on parabola 1 such that y1 = y 
	return (locus.x + x_directrix) / 2 + (locus.y - y) **2 / (2 * (locus.x - x_directrix))


def dist(pt1, pt2):
	return ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5

def translate(pt, displace):
	return pt[0] - displace[0], pt[1] - displace[1]


# Returns True if y is just below parabola 1
def parabolaIntersect(edge, x_directrix, y_pt):


	npOrig = np.array(edge.origin, dtype = "float")
	npDir = np.array(edge.dir, dtype = "float")
	# Value of x and y from t
	x = np.poly1d((edge.dir[0], edge.origin[0]))
	y = np.poly1d((edge.dir[1], edge.origin[1]))

	# Equation of one parabola (below)
	exce = 1/(2*(x_directrix - edge.pt1.x))
	# if ZeroDivisionError, then two points are on the same line
	para = - exce * (y - edge.pt1.y) ** 2 + (x_directrix + edge.pt1.x)/2


	# Equation to solve
	equation = para - x
	disc = equation[1]**2 - 4 * equation[2] * equation[0]

	assert(disc >= 0)

	smallRoot = (-equation[1] - disc ** 0.5) / (2 * equation[2])
	bigRoot = (-equation[1] + disc ** 0.5) / (2 * equation[2])

	assert(smallRoot >= 0 or bigRoot >= 0)

	t = min(t for t in (smallRoot, bigRoot) if t >= 0)

	# Point of intersection of parabolas 
	ptInt = npOrig + t * npDir
	print(ptInt.shape)

	if ptInt[1] == y_pt:
		raise MultipleTargets

	return ptInt[1] > y_pt


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
		#print("Points ({},{}), ({},{}) and ({},{}) are colinear".format(pt1.x, pt1.y, pt2.x, pt2.y, pt3.x, pt3.y))

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
				#print(matrix.T[0])
				#print(origSelf)
				#print(points[0, 0])
				return tuple((origSelf + points[0, 0] * matrix.T[0]).flat)

	def opposite(self):
		return Edge(self.pt1, self.pt2, self.origin, (-self.dir[0], -self.dir[1]))


	def intersect_with_boundary(self, other):

		ptIntersect_t = self.intersect(other)
		ptIntersect = np.array(ptIntersect_t).reshape((2,))
		origSelf = np.array(self.origin).reshape((2,))
		origOther = np.array(other.origin).reshape((2,))

		if self.is_complete():
			boundarySelf = np.array(self.boundary).reshape((2,))
			if np.linalg.norm(ptIntersect - origSelf) > np.linalg.norm(boundarySelf - origSelf):
				raise NoIntersectionBetweenRays

		if self.is_complete():
			boundaryOther = np.array(other.boundary).reshape((2,))
			if np.linalg.norm(ptIntersect - origOther) > np.linalg.norm(boundaryOther - origOther):
				raise NoIntersectionBetweenRays


		return ptIntersect_t






class SearchTree:

	def __init__(self, parent = None, focus = None, children = [], pos = None):

		self.focus = focus
		
		self.pos = pos
		
		self.parent = parent
		self.edge = None

		self.children = children
		self.delete  = []

	def LRiterator(self):
		first = self.leftmost
		while first is not None:
			yield first
			first = first.next

	def leftbranchIterator(self):
		first = self
		while not first.is_leaf():
			yield first
			first = first.leftChild
		else:
			yield first


	def rightbranchIterator(self):
		first = self
		while not first.is_leaf():
			yield first
			first = first.rightChild
		else:
			yield first

	


	def __iter__(self):
		return self.LRiterator()


	def __str__(self):

		# if self.is_leaf():
		# 	return "[focus: {}]".format(self.focus)
		# else:
		# 	return "->".join([str(leaf) for leaf in self])

		def str_rec(tree, n):
			s = n*"\t" + "[parent: {}, focus: {}, edge: {}]\n".format(tree.parent is not None,
															 tree.focus,
															 tree.edge)
			for child in tree.children:
				s += str_rec(child, n+1)

			return s

		# def str_rec(tree, n):
		# 	s = n*"\t" + "[previous: {}, focus: {}, next: {}]\n".format(tree.previous.focus if tree.previous is not None else None,
		# 													 tree.focus,
		# 													 tree.next.focus  if tree.next is not None else None)
		# 	for child in tree.children:
		# 		s += str_rec(child, n+1)

		# 	return s

		return str_rec(self, 0)

	# To disable any event associated with the node
	def setIdleDelete(self):
		for event in self.delete:
			event.setIdle()
		return bool(self.delete)


	@property
	def children(self):
		return self._children
		
	@children.setter
	def children(self, value):
		
		if len(value) == 0:
			self._children = value
			return
		elif len(value) == 2:
			self._children = value
			self.leftChild = value[0]
			self.rightChild = value[1]
		else:
			raise Exception("Non-binary tree modification")


		

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

		#Setting new parent
		value.parent = self
		
	
	
	@property
	def rightChild(self):
		return self.children[1]

	@rightChild.setter
	def rightChild(self, value):
		self.children[1] = value

		#Setting new parent
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
		item = self
		for item in self.leftbranchIterator():
			pass

		return item
	
	@property
	def rightmost(self):
		item = self
		for item in self.rightbranchIterator():
			pass

		return item

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
			try:
				belowLeft = parabolaIntersect(self.edge, x_ligne, y_pt)
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

		print("############### BEFORE COMPILING #########################")
		for edge in self.edges:
			if edge.is_complete():
				print(edge)
			else:
				print("INCOMPLETE EDGE", edge)

		self.compile_edges()

	def compute_graph(self):
		queue = Queue()
		tree = SearchTree(focus = Pt(0, self))
		list_nodes = DoubleList(tree)
		tree.pos = list_nodes

		def check_circle(node, xCurrent):
			left = node.pos.before
			right = node.pos.after

			if left is None or right is None:
				return


			print("Checking {} and {}".format(left.value, right.value))
			
			try:
				pt = x, y = left.value.intersect(right.value)
			except NoIntersectionBetweenRays:
				print("I swear I didn't find anything")
				return
			else:						
				r = dist(node.focus, pt)
				if x+r > xCurrent:
					print("Will occur at x=", x+r)
					#print("Points", node.focus, left.focus, right.focus)
					#print(x,y,r)
					# We store a reference to the delete event
					# If any of these nodes were deleted before the delete event has to take place, 
					# the event has to be disabled
					deleteEvent = Delete(x+r, node)
					deleteEvent.signature = "{} {} {}".format(node.pos.before2.value.focus, node.focus, node.pos.after2.value.focus)
					
					node.delete.append(deleteEvent)
					# node.leftOf.delete.append(deleteEvent)
					# node.rightOf.delete.append(deleteEvent)
					queue.put(deleteEvent)
				else:
					print("I failed")
				

		class Event:
			def __init__(self):
				self.idle = False
			def __lt__(self, other):
				return self.x < other.x

			def __gt__(self, other):
				return self.x > other.x

			def __leq__(self, other):
				return self.x <= other.x

			def __geq__(self, other):
				return self.x >= other.x

			def setIdle(self):
				self.idle = True

			def cond_execute(self):
				if not self.idle:
					print("########################################################")
					self.execute()


		class Delete(Event):

			def __init__(self, x, treeNode):
				self.x = x
				self.treeNode = treeNode
				super(Delete, self).__init__()

			def execute(selfEv):
				print("DELETE ", selfEv.signature, "x=", selfEv.x)
				print("List at delete")
				for thing in list_nodes:
					print(thing)
				
				

				# For adding boundary node to outcoming edge
				right = selfEv.treeNode.pos.after2.value
				left = selfEv.treeNode.pos.before2.value
				assert(right is not None and left is not None)

				print("#DISABLING#")
				if left.setIdleDelete():
					for event in left.delete:
						print("had to disable:", event.signature)
				if right.setIdleDelete():
					for event in right.delete:
						print("had to disable:", event.signature)
				print("END DISABLING")


				focusLeft = left.focus
				focusRight = right.focus
				sca = selfEv.treeNode.leftrightSCA


				# Adding boundaries to collapsing edges
				try:
					edge1 = selfEv.treeNode.pos.before.value
					edge2 = selfEv.treeNode.pos.after.value
					
					intersect = edge1.intersect(edge2)

					edge1.add_boundary(intersect)
					edge2.add_boundary(intersect)
					self.boundaries.append(intersect)
				except AttributeError:
					#print(tree)
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
				dir1 = -(focusLeft.y - focusRight.y), (focusLeft.x - focusRight.x)
				# Direction should be outward from the two edges
				# Inward is when scalar product between the two edges' direction and the new edge direction is negative
				if dir1[0]*edge1.dir[0] + dir1[1] * edge1.dir[1] < 0 and dir1[0]*edge2.dir[0] + dir1[1] * edge2.dir[1] < 0:
					dir1 = -dir1[0], -dir1[1]
				
				edge = Edge(focusLeft, focusRight, origin, dir1)
				self.edges.append(edge)
				sca.edge = edge
				
				# Maintain reference toprevious node
				prevNode = selfEv.treeNode.pos.before2
				prevNode.after.delete(3)
				prevNode.append(edge)

				# Add edge

				#print(tree)

				check_circle(left, selfEv.x)
				check_circle(right, selfEv.x)
				

		class Insert(Event):
			def __init__(self, focus):

				# self.focus is instance Pt
				self.focus = focus
				self.x = self.focus.x
				super(Insert, self).__init__()

			def execute(selfEv):
				
				try:
					node, xIntersect = tree.find(selfEv.x, selfEv.focus.y)
				except MultipleTargets as t:
					raise Exception("Multiple target found ; not implemented yet")
				else:
					print("INSERT", selfEv.focus.idx,"in", node.focus, "x=", selfEv.x)
					print("List at insert")
					for thing in list_nodes:
						print(thing)
					
					# getting ref to position in list
					pos = node.pos
					assert(pos is not None)

					# creating new right child of "node"
					st = SearchTree(children = [SearchTree(focus = selfEv.focus), SearchTree(focus = node.focus)])

					# Sprouting new children
					node.children = [SearchTree(focus = node.focus, pos = pos), st]

					# Augment list of node with inserted nodes
					pos.value = node.leftChild
					pos.append(st.rightChild)
					pos.append(st.leftChild)

					# Add references in trees
					st.leftChild.pos = pos[1]
					st.rightChild.pos = pos[2]



					# Inner nodes have no position in list
					node.pos = None

					# Resetting delete events
					print("#DISALING#")
					if node.setIdleDelete():
						for event in node.delete:
							print("had to disable:", event.signature)
					print("#END DISABLING#")

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

					# Add edges to pos
					node.leftChild.pos.append(edge)
					st.leftChild.pos.append(opEdge)

					self.edges.append(opEdge)
					self.edges.append(edge)
					
					# Delete focus for inner node
					node.focus = None

					# Check for circle intersection
					check_circle(st.rightChild, selfEv.x)
					check_circle(node.leftChild, selfEv.x)

					#print("Output insert:")
					#print(tree)

				

		for i in range(len(self.pts) - 1, 0, -1):
			queue.put(Insert(Pt(i, self)))

		for event in queue:
			event.cond_execute()
				
	def compile_edges(self):

		# Sort by lexicographic order so that edges adjacent in "self.edges" are between the same cells
		def lexi(edge):
			return edge.pt1.idx, edge.pt2.idx
		self.edges.sort(key = lexi)

		def adjacent(iterator):
			current = next(iterator)
			while True:
				try:
					nextV = next(iterator) 
				except StopIteration:
					yield (current,)
					break
				else:
					if lexi(current) == lexi(nextV):
						yield current, nextV
						try:
							current = next(iterator)
						except StopIteration:
							break
					else:
						yield (current,)
						current = nextV
				

		new_edges = []
		for t in adjacent(iter(self.edges)):
			if len(t) == 1:
				# edge = t[0]
				# toAppend = Edge(edge.pt1, edge.pt2, edge.boundary, edge.dir)
				new_edges.append(t[0])
			else:
				edge1, edge2 = t
					
				toAppend = Edge(edge1.pt1, edge1.pt2, edge1.origin, edge1.dir)

				if edge1.is_complete() and edge2.is_complete():
					toAppend.origin = edge1.boundary
					toAppend.dir = edge2.dir
					toAppend.add_boundary(edge2.boundary)
				elif edge1.is_complete():
					toAppend.origin = edge1.boundary
					toAppend.dir = edge2.dir
				else:
					toAppend.origin = edge2.boundary
					toAppend.dir = edge1.dir

				new_edges.append(toAppend)

		self.edges = new_edges

	def draw_img(self, size, thick = 1, center = None, mode = "RGB"):
		h, w = size

		if center is None:
			center = (0., 0.)

		img = Image.new(mode, size, (0,0,0))
		draw = ImageDraw.Draw(img)

		
		for edge in self.edges:

			if edge.is_complete():
				origin = translate(edge.origin, center)
				end = translate(edge.boundary, center)
				draw.line([origin, end], width = thick, fill = (255,255,255))
			else:
				npOrigin = np.array(edge.origin)
				npDir = np.array(edge.dir)
				npDir = npDir/np.linalg.norm(npDir)

				endPt = npOrigin + (h+w) * npDir
				endPt = endPt[0], endPt[1]
				endPt = translate(endPt, center)

				origin = translate(edge.origin, center)

				draw.line([origin, endPt], width = thick, fill = (255,255,255))
		return img

	#def draw_cell(self, size, thick = 1, center = None)
				




if __name__ == "__main__":
	a = Edge(0,1,(0,0),(1,1))
	b = Edge(0,1,(0,1),(2,-1))
	#a.intersect(b)

	scale = 50
	# testPts = [[5.80422285e+01, 9.70424354e+01],
 #       [2.45994724e+02, 2.15062901e+02],
 #       [8.91472245e+01, 2.20015415e+02],
 #       [1.30929163e+02, 2.59899312e+02],
 #       [2.33091249e+02, 2.32009340e+02],
 #       [2.90028884e+02, 2.52980960e+02],
 #       [1.70709103e+01, 1.01227005e+01],
 #       [1.64920528e+02, 3.96799467e+00],
 #       [2.48008720e+02, 1.75997180e+02],
 #       [1.15334630e-02, 5.29805219e+01],
 #       [6.60150582e+01, 2.67972627e+02],
 #       [2.43987617e+02, 2.19658159e+01],
 #       [2.83983344e+02, 1.75926724e+02],
 #       [2.84097323e+02, 1.07987682e+02],
 #       [1.47980174e+02, 2.82042400e+02],
 #       [1.93032889e+02, 2.16995710e+02],
 #       [2.04012869e+02, 2.50453043e+01],
 #       [2.09052708e+02, 2.96915271e+02],
 #       [1.87993088e+02, 8.20256878e+01],
 #       [5.40024820e+01, 2.70968975e+02]]
	testPts = [[ 43,  77],
       [262, 230],
       [298, 251],
       [161, 229],
       [145,  95]]
	voronoi = VoronoiGraph(testPts)
	print("############### AFTER COMPILING #########################")
	for edge in voronoi.edges:
		if edge.is_complete():
			print(edge)
		else:
			print("INCOMPLETE EDGE", edge)


	


