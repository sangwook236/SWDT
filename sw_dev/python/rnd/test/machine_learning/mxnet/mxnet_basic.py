#import mxnet.ndarray as nd
from mxnet import nd
from mxnet import autograd

# REF [site] >> https://gluon-crash-course.mxnet.io/ndarray.html
def ndarray_example():
	a = nd.array(((1, 2, 3), (5, 6, 7)))
	b = nd.full((2, 3), 2.0)
	b.shape, b.size, b.dtype

	# Operations.
	x = nd.ones((2, 3))
	y = nd.random.uniform(-1, 1, (2, 3))
	x * y
	y.exp()
	nd.dot(x, y.T)

	# Indexing.
	y[1, 2]
	y[:, 1:3]
	y[:, 1:3] = 2
	y[1:2, 0:2] = 4

	# Converting between MXNet NDArray and NumPy.
	na = x.asnumpy()
	nd.array(na)

# REF [site] >> https://gluon-crash-course.mxnet.io/autograd.html
def autograd_example():
	# When differentiating a function f(x)=2x2 with respect to parameter x.
	x = nd.array([[1, 2], [3, 4]])
	x.attach_grad()

	# To let MXNet store y, so that we can compute gradients later, we need to put the definition inside a autograd.record() scope.
	with autograd.record():
		y = 2 * x * x

	# Invoke back propagation (backprop).
	#	When y has more than one entry, y.backward() is equivalent to y.sum().backward().
	y.backward()

	print('x.grad =', x.grad)

	# Using Python control flows.
	def f(a):
		b = a * 2
		while b.norm().asscalar() < 1000:
			b = b * 2
		if b.sum().asscalar() >= 0:
			c = b[0]
		else:
			c = b[1]
		return c

	a = nd.random.uniform(shape=2)
	a.attach_grad()

	with autograd.record():
		c = f(a)
	c.backward()

def main():
	ndarray_example()
	autograd_example()

#%%-------------------------------------------------------------------

if '__main__' == __name__:
	main()
