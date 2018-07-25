from ipywidgets import Output
from cst import *
import jupyter.updater as updater

def update_val(name, value, change):
	"""Updates parameter of operator and refreshes display. Meant to be used as observer function."""
	updater.u[name][0][value] = change["new"]
	updater.u.refresh()


def opDisplay(op, active = D_ACTIVE, name = None):
	"""Creates and registers an operator display output."""

	if name is None:
		# Assign oneself a custom name of the form "flatN" where N is number
		name = customName("op", updater.u)

	output = Output()
	# display(output)
	with output:
		display(op.process())

	updater.u.add(name, op, output, active)

	return output


