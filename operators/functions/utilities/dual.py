import numpy as np
from periodic_areas import * 

# Constructs dual of periodic triangulation
def dual(graph):

	edges = graph.edges

	# Create areas holder for edges
	for edge in edges:
		edge.areas = [] # RArea

	# Create areas

	for edge in edges:
		if len(edge.areas) == 2:
			continue
		for i,(link1, link2) in enumerate(edge.common_links()):
			print(i)
			rv1 = link1.other.r(graph.toSq, edge.origin.rep + link1.offset)
			if len(edge.areas) == 0 or (not (edge.areas[0].connects(edge.origin, edge.end, rv1))):

				# Create area from verts
				area = Area(pts = [edge.origin, edge.end, rv1], edges = [link1.edge, link2.edge, edge])
				edge.areas.append(area)
				link1.edge.areas.append(area)
				link2.edge.areas.append(area)

	# Join areas