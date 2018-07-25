class Op:
	"""
	This class implements the general logic of operators

	An operator has the following main elements:
		- *params* : parameter inputs which may themselves be operators
		- *function* : the function it computes from these inputs

	The method *process()* returns the output of the operator.
	This function is lazy and only computes when changes has been made to the input.

	To obtain this lazy behaviour, operators are endowed with a *change* property.
	When set to True, this property toggles the change property of all operators this operator is input to to True. 
	"""
	def __init__(self, function, **params):
		self.output = None
		self.parents = []
		self.change = True
		self.params = params
		self.function = function

		for key, value in self.params.items():
			if isinstance(value, Op):
				value.register_parent(self)
		
	def process(self):
		if self.change:
			dRplOp = {key: value if not isinstance(value, Op) else value.process() for key, value in self.params.items()}
			self.output = self.function(**dRplOp)
			self.change = False
			return self.output
		else:
			return self.output

	def register_parent(self, parent):
		if all(p is not parent for p in self.parents):
			self.parents.append(parent)

	def unregister_parent(self, parent):
		for i, p in enumerate(self.parents):
			if p is parent:
				self.parents.pop(i)

	def get_change(self):
		return self._change

	def set_change(self,value):
		self._change = value
		if value:
			for parent in self.parents:
				parent.change = True

	change = property(get_change, set_change)

	def __getitem__(self, key):
		return self.params[key]

	def __setitem__(self, key, value):
		
		if isinstance(self.params[key], Op):
			self.params[key].unregister_parent(self)
		if isinstance(value, Op):
			value.register_parent(self)

		self.change = True
		self.params[key] = value


# Example Operators for test
def spatzle(a,b):
	return a + b
op1 = Op(spatzle, a = 1, b = 2)
op2 = Op(spatzle, a = 3, b = 4)
op3 = Op(spatzle, a = op1, b = op2)