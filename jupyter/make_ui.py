from IPython.display import display, clear_output
from operators.flat import Flat
import ipywidgets as widgets
from jupyter.ui_common import colorSlider
from utils import customName
from jupyter.ui_init import opDisplay, update_val
from cst import *
import jupyter.updater as updater


def createUIforOP(op, dWidgets, name = None, prefix = "op"):

	if name is None:
		name = customName(prefix, updater.u)


	# Creating and displaying left panel
	outputParams = widgets.Output()
	
	with outputParams:
		print("Name: ", name)
		for key, val in dWidgets.items():
			display(val)

	# Show image
	out = opDisplay(op, True, name)

	box = widgets.HBox([outputParams, out])
	box.layout.align_items = 'center'
	display(box)


	# Assign events on change

	for key, val in dWidgets.items():
		print(key)
		val.observe(lambda change, n = name, k = key: update_val(n, k, change), names = "value")
	# Careful: we need the "name" variable to be bound locally ; otherwise, future changes to "name" result in change of the event function itself!
	# Hence the default parameter trick
