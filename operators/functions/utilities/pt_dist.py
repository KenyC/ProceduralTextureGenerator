import numpy as np
from math import sqrt

HEIGHT_TRI = sqrt(3)/2



def triangular_range(n_x, n_y = None, step_x = None, step_y = None, center = None):
	
	if center is None:
		center = np.full(2, 0.)

	if step_y is None:
		step_y = step_x * HEIGHT_TRI

	if step_x is None:
		step_x = step_y / HEIGHT_TRI

	if n_y is None:
		n_y = int((n_x * step_x // (2 * step_y)) * (2 * step_y) / step_x)

	# Constructing points
	# We first build a rectangular grid

	xs = np.arange(n_x) * step_x # shape: N_X
	ys = np.arange(n_y) * step_y # shape: N_Y

	xv, yv = np.meshgrid(xs, ys) # shape of each : N_X, N_Y
	points = np.stack((xv, yv), axis = 2) # shape N_X, N_Y, 2

	# We now shift odd rows by half a step
	points[1::2, :, 0] += step_x / 2

	# We flatten the first two axis to get a list of points
	points = np.reshape(points, (n_x * n_y, 2)) + center

	return points

def triangular_width(n_x, width, center = None):
	return triangular_range(n_x, step_x = width / n_x, center = center)

def jitter(pts, scale):
	return pts + np.random.normal(scale = scale, size = pts.shape)