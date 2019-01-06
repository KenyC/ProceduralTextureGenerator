class DoubleList:

	def fromList(l):
		toReturn = DoubleList(l[0])

		for x in reversed(l[1:]):
			toReturn.append(x)

		return toReturn

	def __init__(self, value, after = None, before = None):
		self.before = before
		self.after = after
		self.value = value

	def delete(self, nb = 1):
		if nb == 1:
			self.after.before = self.before
			return self.after
		else:
			val = self
			for i in range(nb):
				val = val.delete()
			return val

	# Appends after node
	def append(self, value):
		dl = DoubleList(value, self.after) # self.after now points to "dl"
		self.after = dl # dl now points to self

	# Inserts dl list after
	def insert(self, value):
		value.last.after = self.after
		self.after = value

	@property
	def after2(self):
		return self.after.after

	@property
	def before2(self):
		return self.before.before
	
	

	@property
	def last(self):
		toReturn = self
		for toReturn in self.LtoRIterator():
			pass
		return toReturn

	@property
	def first(self):
		toReturn = self
		for toReturn in self.RtoLIterator():
			pass
		return toReturn
	
	def __getitem__(self, idx):
		if isinstance(idx, int):
			toReturn = self
			for toReturn, _ in zip(iter(self), range(idx + 1)):
				pass
			return toReturn
		else:
			raise KeyError

	@property
	def before(self):
		return self._before
	 
	@before.setter
	def before(self, value):
		self._before = value
		if value is not None:
			value._after = self


	@property
	def after(self):
		return self._after
	 
	@after.setter
	def after(self, value):
		self._after = value
		if value is not None:
			value._before = self

	def LtoRIterator(self):
		first = self
		while first is not None:
			yield first
			first = first.after

	def RtoLIterator(self):
		first = self
		while first is not None:
			yield first
			first = first.before


	def __iter__(self):
		return self.LtoRIterator()

	def __str__(self):
		return " <-> ".join([str(dl.value) for dl in self])


if __name__ == "__main__":

	test1 = DoubleList(35)

	for i in range(5):
		test1.append(i)
	# Should be 35, 4, 3, 2, 1, 0

	test2 = DoubleList(27)
	for i in range(3):
		test2.append(i)

	test1[3].insert(test2)

	for dl in test1:
		print(dl.value)

	# Ensure that references are well guarded
	l = [[0], [52], [21]]
	test3 = DoubleList.fromList(l)
	test3.value[0] = 21
	print(l)
