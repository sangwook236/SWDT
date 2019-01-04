#!/usr/bin/env python

# REF [site] >> https://www.mathworks.com/help/matlab/matlab-engine-for-python.html

import matlab.engine

# REF [site] >> https://www.mathworks.com/help/matlab/matlab_external/use-the-matlab-engine-workspace-in-python.html
def basic(engine):
	a, b = 1.0, 2.0
	engine.workspace['a'] = a
	engine.workspace['b'] = b
	#engine.eval('c = a + b;')  # Error.
	engine.eval('c = a + b;', nargout=0)
	c = engine.workspace['c']
	print('{} + {} = {}'.format(a, b, c))

	x = 4.0
	engine.workspace['y'] = x
	a = engine.eval('sqrt(y)')
	print('sqrt({}) = {}'.format(x, a))

	a = matlab.double([1, 4, 9, 16, 25])
	b = engine.sqrt(a)
	print('sqrt({}) = {}'.format(a, b))

# REF [site] >> https://www.mathworks.com/help/matlab/matlab_external/call-matlab-functions-from-python.html
def call_matlab_function(engine):
	tf = engine.isprime(37)
	print('Is prime =', tf)

	t = engine.gcd(100.0, 80.0, nargout=3)
	print('GCD =', t)

# REF [site] >> https://www.mathworks.com/help/matlab/matlab_external/call-user-script-and-function-from-python.html
def call_user_function(engine):
	# User script.
	engine.workspace['b'] = 5.0
	engine.workspace['h'] = 3.0
	engine.triarea_script(nargout=0)
	area = engine.workspace['a']
	print('Area =', area)

	# User function.
	area = engine.triarea_func(1.0, 5.0)
	print('Area =', area)

def main():
	engine = matlab.engine.start_matlab()

	basic(engine)

	call_matlab_function(engine)
	call_user_function(engine)

	engine.quit()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
