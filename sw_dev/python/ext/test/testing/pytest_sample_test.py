from add import add
from subtract import subtract

def test_add():
	assert add(1, 3) == 5

# NOTE [info] >> Is not tested.
def add_test():
	assert add(1, 3) == 5

def test_subtract():
	assert subtract(5, 3) == 3

# NOTE [info] >> Is not tested.
def subtract_test():
	assert subtract(5, 3) == 3
