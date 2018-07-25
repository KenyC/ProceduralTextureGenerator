from PIL.ImageChops import add
from op import Op

class Add(Op):
	def __init__(self, **params):
		super(Add, self).__init__(add, **params)