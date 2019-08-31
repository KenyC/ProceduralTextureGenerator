from scipy.spatial import Voronoi
from PIL import Image, ImageDraw
import numpy as np

rotation = np.array([[0, 1], [-1, 0]], dtype = "float")

class Edge:
	def __init__(self, pt1, pt2, origin, end = None, direction = None):
		self.pt1 = pt1 
		self.pt2 = pt2
		
		self.origin = origin
		
		if direction is None and end is not None:
			self.end = end
			self.direction = self.end - self.origin
			self.bounded = True
		else:
			self.direction = direction
			self.bounded = False

class Cell:
	def __init__(self, point, edges):
		self.pt = point
		self.edges = edges

class VoronoiGraph(Voronoi):

	def __init__(self, *args, **kwargs):
		super(VoronoiGraph, self).__init__(*args, **kwargs)
		self.compile_edges()
		self.compile_cells()
		
	def compile_cells(self):
		self.cells = [Cell(pt, []) for pt in self.points]

		for edge in self.edges:
			self.cells[edge.i_pt1].edges.append(edge)
			self.cells[edge.i_pt2].edges.append(edge)


	def compile_edges(self):
		self.edges = []
		center_of_mass = np.mean(self.points, axis = 0)
		
		for (idx_pt1, idx_pt2), (idx1, idx2) in zip(self.ridge_points, self.ridge_vertices):
			
			pt1 = self.points[idx_pt1]
			pt2 = self.points[idx_pt2]
			
			if idx1 != -1 and idx2 != -1:
				origin = self.vertices[idx1] 
				end = self.vertices[idx2] 

				e = Edge(pt1, pt2, origin, end = end)
				e.i_pt1 = idx_pt1
				e.i_pt2 = idx_pt2

				self.edges.append(e)
			else:

				if idx1 == -1:
					origin = self.vertices[idx2]
				else:
					origin = self.vertices[idx1]

				direction = rotation.dot(pt2 - pt1)
				direction = direction / np.linalg.norm(direction)
				
				middle = (pt2 + pt1) / 2
				# To determine the direction of infinite edge, rely on the fact that it will be outgoing from the center of mass
				if np.dot(middle - center_of_mass, direction) < 0:
					direction = -direction

				e = Edge(pt1, pt2, origin, direction = direction)

				e.i_pt1 = idx_pt1
				e.i_pt2 = idx_pt2
				
				self.edges.append(e)
		
		
	def draw_img(self, size, thick = 1, center = None, mode = "RGB", display_points = False):
		h, w = size

		if center is None:
			center = (0., 0.)

		img = Image.new(mode, size, (0,0,0))
		draw = ImageDraw.Draw(img)
		
		for edge in self.edges:
			if edge.bounded:
				draw.line([tuple(edge.origin), tuple(edge.end)], width = thick, fill = (255, 255, 255))
			else:
				draw.line([tuple(edge.origin), tuple(edge.origin + (h + w) * edge.direction)], width = thick, fill = (255, 255, 255))

		radius = 3

		if display_points:
			for pt in self.points:
				top_left = tuple(pt - radius)
				bot_right = tuple(pt + radius)
				draw.ellipse([top_left, bot_right], fill = (255, 0, 0))
		
		return img
		
def fill_img(self, size, center = None, mode = "RGB"):
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

    for cell in self.cells:

        cell.mask = np.full_like(img, 1.)
        cell.dists = []

        for edge in cell.edges:

            # Determining origin and inward-pointing normal of edge
            edge.origin = np.array(edge.origin)
            edge.normal = rotation.dot(edge.direction)
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
    for cell in self.cells:

        for edge, dist in zip(cell.edges, cell.dists):
            # Multiplying positive value of plane
            img *= cell.mask * dist + (1 - cell.mask) 

        # Cell-wise renormalization
        maximum = np.max(cell.mask * img)
        img /= cell.mask * maximum + (1 - cell.mask)

    return img


if __name__ == "__main__":
	N = 4
	size = height, width = 300 ,300
	pts = np.stack((np.random.randint(0, 300, N), np.random.randint(0, 300, N)), axis = -1)

	voronoi = VoronoiGraph(pts)
	img = voronoi.draw_img(size, display_points = True)