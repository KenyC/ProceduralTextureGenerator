import numpy as np
from voronoi_utils import VoronoiGraph, Edge, NoIntersectionBetweenRays, translate, dist
# This file computes a voronoi image with continuous colours where points are colored in white and edges in black


class Cell:

	def __init__(self, pt, edges = None):
		self.pt = pt
		if edges is None:
			self.edges = []
		else:
			self.edges = edges

	def contains(self, pt):
		ray = Edge(None, None, pt, (1,0))

		nIntersect = 0
		for edge in self.edges: 
			try:
				ray.intersect_with_boundary(edge)
			except NoIntersectionBetweenRays:
				pass
			else:
				nIntersect += 1

		# There is a hairy case to implement: what to do if one of the cell has a wall at infinity?

		return (nIntersect%2 == 1) # If a ray to infinity intersects the polygon's boundary an

def dist_to_edge(pt, edge):
	pt = np.array(pt)
	orig = np.array(edge.origin)
	direction = np.array(edge.dir)

	return np.cross(pt - orig, direction) / np.linalg.norm(direction)

def plane(origin, normal, coords):
	dot_product = np.dot(coords - origin, normal) > 0
	
	plane = np.full_like(dot_product, 0.)
	plane[dot_product] = 1.
	
	return plane

class VoronoiFill(VoronoiGraph):

	def compute_cells(self):

		for i, pt in enumerate(self.pts):
			self.pts[i] = tuple(pt)

		self.cells = {pt: Cell(pt) for pt in self.pts}

		for edge in self.edges:
			self.cells[self.pts[edge.pt1.idx]].edges.append(edge)
			self.cells[self.pts[edge.pt2.idx]].edges.append(edge)


	def fill_img(self, size, thick = 1, center = None, mode = "RGB"):

		h, w = size

		if center is None:
			center = np.array([0.,0.])

		img = np.full((h, w), 1.)

		# We create a coordinate array
		# Shape: w
		x = np.arange(w) + 0.5 - center[0]
		# Shape: h
		y = np.arange(h) + 0.5 - center[1]
		# Shapes: (w, h)
		xv, yv = np.meshgrid(x, y)

		# coordinate array
		# Shape: (w, h, 2)
		coords = np.stack((xv, yv), axis = 2)



		# Constructing masks for every cell

		for cell in self.cells.values():

			cell.mask = np.full_like(img, 1.)
			cell.dists = []
			
			for edge in cell.edges:

				# Determining origin and inward-pointing normal of edge
				edge.origin = np.array(edge.origin)
				edge.normal = np.array([-edge.dir[1], edge.dir[0]])
				if np.dot(edge.normal, np.array(cell.pt) - edge.origin)  < 0:
					normal = - edge.normal
				else:
					normal = edge.normal

				# Multiplying positive value of plane
				dist = np.dot(coords - edge.origin, normal) / np.linalg.norm(normal)
				cell.dists.append(dist)

				mult = (dist >= 0).astype("float")
				cell.mask *= mult

		# Multiplying distances
		for cell in self.cells.values():
			
			for edge, dist in zip(cell.edges, cell.dists):
				# Multiplying positive value of plane
				img *= cell.mask * dist + (1 - cell.mask) 

			# Cell-wise renormalization
			maximum = np.max(cell.mask * img)
			img /= cell.mask * maximum + (1 - cell.mask)

		return img

if __name__ == "__main__":
	N = 20
	length = 900
	size = length ,length
	pts = np.stack((np.random.randint(0, length, N), np.random.randint(0, length, N)), axis = -1)
	# Jitter to avoid degenerates
	pts = pts +  np.random.normal(scale = 1/N, size = pts.shape)

	voronoi = VoronoiFill(pts.tolist())
	voronoi.compute_cells()

	img =  voronoi.draw_img(size)