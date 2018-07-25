import ipywidgets as widgets

class colorSlider(widgets.IntSlider):
	def __init__(self, name):
		super(colorSlider, self).__init__(
				value=128,
				min=0,
				max=255,
				step=1,
				description=name,
				disabled=False,
				continuous_update=False,
				orientation='horizontal',
				readout=True,
				readout_format='d'
				)

class powerTwoSlider(widgets.SelectionSlider):
	def __init__(self, name, min = 1, max = 13):
		super(powerTwoSlider, self).__init__(
				options = [2**i  for i in range(min,max)],
				value = 2**min,
				description = name,
				disabled=False,
				continuous_update=False,
				orientation='horizontal',
				readout=True
				)

