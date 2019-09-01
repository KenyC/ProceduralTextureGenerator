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

	def overlap(self, size, center):
		size = np.array(size)
		center = np.array(center)

		for edge in self.edges:
			vertices = [edge.origin, edge.end] if edge.bounded else [edge.origin]
			for pt in vertices:
				translated = (pt - center)
				if np.all(translated <= size) and np.all(translated >= 0):
					return True
		return False

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
		w, h = size

		if center is None:
			center = (0., 0.)

		img = Image.new(mode, size, (0,0,0))
		draw = ImageDraw.Draw(img)
		
		for edge in self.edges:
			if edge.bounded:
				draw.line([tuple(edge.origin - center), tuple(edge.end - center)], width = thick, fill = (255, 255, 255))
			else:
				draw.line([tuple(edge.origin - center), tuple(edge.origin - center + (h + w) * edge.direction)], width = thick, fill = (255, 255, 255))

		radius = 3 

		if display_points:
			for pt in self.points:
				top_left = tuple(pt - radius - center)
				bot_right = tuple(pt + radius - center)
				draw.ellipse([top_left, bot_right], fill = (255, 0, 0))
		
		return img
		

	def fill_img(self, size, center = None, mode = "RGB"):

		cells = [cell for cell in self.cells if cell.overlap(size, center)]

		size_img = size
		w, h = size_img

		img = np.full(size_img, 0.)

		for cell in cells:
		    
		    all_edges = np.stack([edge.origin for edge in cell.edges] + [edge.end for edge in cell.edges if edge.bounded])

		    minPt = np.min(all_edges, axis = 0)
		    maxPt = np.max(all_edges, axis = 0)
		    
		    cell.iMin = (minPt - center).astype("int")
		    cell.iMax = (maxPt - center).astype("int")
		    cell.size_img = cell.iMax - cell.iMin
		    cell.img = np.full(cell.size_img, 1.)
		    
		    # We create a coordinate array
		    # Shape: w
		    x = np.arange(cell.size_img[0]) + 0.5 + minPt[0] 
		    # Shape: h
		    y = np.arange(cell.size_img[1]) + 0.5 + minPt[1] 
		    # Shape: (w, h, 2)
		    coords = np.stack(np.meshgrid(x, y, indexing = "ij"), axis = 2)

		    for edge in cell.edges:

		        # Determining origin and inward-pointing normal of edge
		        edge.normal = rotation.dot(edge.direction)
		        if np.dot(edge.normal, cell.pt - edge.origin)  < 0:
		            normal = - edge.normal
		        else:
		            normal = edge.normal
		        
		        # Calculating distance to edge
		        dist = np.dot(coords - edge.origin, normal) / np.linalg.norm(normal)
		        
		        # Multiplying positive value of plane
		        cell.img *= dist * (dist >= 0).astype("float")

		    cell.img /= np.max(cell.img)

		# Multiplying distances
		for cell in cells:
		    iMin = np.maximum(cell.iMin, [0, 0]).astype("int")
		    iMax = np.minimum(cell.iMax, size_img).astype("int")

		    cropping_left = iMin - cell.iMin
		    cropping_right = cell.size_img - (cell.iMax - iMax)
		    img[iMin[0]:iMax[0], iMin[1]:iMax[1]] += cell.img[cropping_left[0]:cropping_right[0],
		    												cropping_left[1]:cropping_right[1]]

		return img

if __name__ == "__main__":
	N = 20
	size = height, width = 300 ,300
	pts = np.stack((np.random.randint(0, 300, N), np.random.randint(0, 300, N)), axis = -1)

	voronoi = VoronoiGraph(pts)
	img = voronoi.draw_img(size, display_points = True)