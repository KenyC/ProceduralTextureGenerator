from IPython.display import display, clear_output
from cst import *


def redraw(outlet, img):
	""" Clear and display output process"""
	with outlet:
		clear_output(wait = True)
		display(img)
		pass


class Updater:
	""" This class listens to parameters update and redraw the active cells"""

	def __init__(self):
		self.out = {}

	def add(self, key, op, out, active = D_ACTIVE):
		self.out[key] = (op, out, active)

	def delete(self, key):
		self.out.pop(key, None)

	def __contains__(self, key):
		return (key in self.out)

	def __getitem__(self, key):
		return self.out[key]

	def __setitem__(self, key, value):
		self.out[key] = value

	#Refresh displays
	def refresh(self):
		for key, val in self.out.items():
			
			op, out, active = val
			if active:
				redraw(out, op.process())

u = Updater()