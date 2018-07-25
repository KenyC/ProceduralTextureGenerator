Caveats
-----------------------------

Size of an image is always a tuple (width, height). Similarly, coordinates (x,y) represent horizontal-vertical. (0,0) is upper-left corner.

Updater's dictionnary (jupyter/update.py) stores triples (operator, display output, is active?)

Run code from top directory ; includes are tailored to that particular location


Plot line
----------------------------

A Jupyter notebook with custom widgets defining each operator.

Operators are defined by parameters that can themselves be other operators. Changes to one parameter and refreshing is optimal ; only the affected bits are recalculated.

