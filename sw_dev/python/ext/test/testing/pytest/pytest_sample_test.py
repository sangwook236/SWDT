from add import add
from subtract import subtract

def test_add():
	assert add(1, 3) == 5

# NOTE [info] >> Will be not tested.
def add_test():
	assert add(1, 3) == 5

def test_subtract():
	assert subtract(5, 3) == 3

# NOTE [info] >> Will be not tested.
def subtract_test():
	assert subtract(5, 3) == 3

class TestArithmetic(object):
	def test_add(self):
		assert add(1, 3) == 5

	# NOTE [info] >> Will be not tested.
	def add_test(self):
		assert add(1, 3) == 5

	def test_subtract(self):
		assert subtract(5, 3) == 3

	# NOTE [info] >> Will be not tested.
	def subtract_test(self):
		assert subtract(5, 3) == 3

class testArithmetic(object):
	def test_add(self):
		assert add(1, 3) == 5

	# NOTE [info] >> Will be not tested.
	def add_test(self):
		assert add(1, 3) == 5

	def test_subtract(self):
		assert subtract(5, 3) == 3

	# NOTE [info] >> Will be not tested.
	def subtract_test(self):
		assert subtract(5, 3) == 3

class ArithmeticTest(object):
	def test_add(self):
		assert add(1, 3) == 5

	# NOTE [info] >> Will be not tested.
	def add_test(self):
		assert add(1, 3) == 5

	def test_subtract(self):
		assert subtract(5, 3) == 3

	# NOTE [info] >> Will be not tested.
	def subtract_test(self):
		assert subtract(5, 3) == 3
