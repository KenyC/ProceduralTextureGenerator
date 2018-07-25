from IPython.display import display, clear_output
from operators.flat import Flat
import ipywidgets as widgets
from ui_common import colorSlider
from jupyter.make_ui import createUIforOP
from cst import *
from utils import customName
import jupyter.updater as updater

actual = Flat(GRAY)

createUIforOP(actual, {"red": colorSlider("Red"),
					"green": colorSlider("Green"),
					"blue": colorSlider("Blue")
			})


# # Assign oneself a custom name of the form "flatN" where N is number
# name = customName("flat", updater)

# # Creating operator
# actual = Flat(GRAY)

# # Creating and displaying slider
# outputParams = Output()
# redSlider = colorSlider("Red")
# greenSlider = colorSlider("Green")
# blueSlider = colorSlider("Blue")

# with outputParams:
# 	print("Name: ", name)
# 	display(redSlider)
# 	display(greenSlider)
# 	display(blueSlider)

# # Show image
# out = opDisplay(actual, True, name)

# box = widgets.HBox([outputParams, out])
# box.layout.align_items = 'center'
# display(box)


# # Assign event on change
# redSlider.observe(lambda change, n = name: update_val(n, "red", change), names = "value")
# greenSlider.observe(lambda change, n = name: update_val(n, "green", change), names = "value")
# blueSlider.observe(lambda change, n = name: update_val(n, "blue", change), names = "value")
# # Careful: we need the "name" variable to be bound locally ; otherwise, future changes to "name" result in change of the event function
# # Hence the default parameter trick
