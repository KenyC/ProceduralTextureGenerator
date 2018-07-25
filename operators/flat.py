from op import Op
from operators.functions.flat import flat
from cst import *


class Flat(Op):
	def __init__(self, color = D_COL, size = D_SIZE, mode = D_MODE):
		super(Flat, self).__init__(flat, color = color, size = size, mode = mode)
		
	def set_red(self,value):
		self["color"] = value, self["color"][1], self["color"][2]

	def get_red(self):
		return self["color"][0]

	red = property(get_red, set_red)

	def set_green(self,value):
		self["color"] = self["color"][0], value,  self["color"][2]

	def get_green(self):
		return self["color"][1]

	green = property(get_green, set_green)

	def set_blue(self,value):
		self["color"] = self["color"][0], self["color"][1], value

	def get_blue(self):
		return self["color"][2]

	blue = property(get_blue, set_blue)

	def __getitem__(self, key):
		if key == "red":
			return self.red
		elif key == "green":
			return self.green
		elif key == "blue":
			return self.blue
		else:
			return super(Flat, self).__getitem__(key)

	def __setitem__(self, key, value):
		self.change = True

		if key == "red":
			self.red = value
		elif key == "green":
			self.green = value
		elif key == "blue":
			self.blue = value
		else:
			super(Flat, self).__setitem__(key, value)

