#!/usr/bin/env python

def args_func(*args):
	print('********** args_func()')
	# NOTE [info] >> args is not sorted.
	print('{}: {}'.format(args, type(args)))
	if args is not None:
		for arg in args:
			print('\t{}'.format(arg))

	#args_func2(args)
	args_func2(*args)
	# NOTE [error] >> TypeError: kwargs_func2() argument after ** must be a mapping, not tuple.
	#kwargs_func2(**args)

def kwargs_func(**kwargs):
	print('********** kwargs_func()')
	# NOTE [info] >> kwargs is sorted. (???)
	print('{}: {}'.format(kwargs, type(kwargs)))
	if kwargs is not None:
		for key, value in kwargs.items():
			print('\t{} = {}'.format(key, value))

	args_func2(*kwargs)
	#args_func2(*kwargs.keys(), *kwargs.values())
	args_func2(*kwargs.keys())
	args_func2(*kwargs.values())
	kwargs_func2(**kwargs)

def args_func2(first_arg, *args2):
	print('********** args_func2()')
	print('1st arg = {}'.format(first_arg))
	print('{}: {}'.format(args2, type(args2)))

# NOTE [error] >> TypeError: kwargs_func2() missing 1 required positional argument: 'first_arg'.
#def kwargs_func2(first_arg, param1=None, param2=None, param3=None, **kwargs2):
def kwargs_func2(last_name, param1=None, param2=None, param3=None, **kwargs2):
	print('********** kwargs_func2()')
	print('1st arg = {}'.format(last_name))  # last_name is a positional argument.
	print('param1 = {}, param2 = {}'.format(param1, param2))
	print('{}: {}'.format(kwargs2, type(kwargs2)))

def args_kwargs_func(*args, **kwargs):
	args_func(678, *args, 'XyZ')
	kwargs_func(grade=2, **kwargs, school='elementary')

	# NOTE [error] >> args_func() got an unexpected keyword argument 'school'.
	#args_func(678, *args, 'XyZ', grade=2, school='elementary')
	# NOTE [error] >> kwargs_func() takes 0 positional arguments but 2 were given.
	#kwargs_func(678, 'XyZ', grade=2, **kwargs, school='elementary')

def main():
	# NOTE [error] >> SyntaxError: positional argument follows keyword argument.
	#args_kwargs_func(1, 3.7, 'abc', first_name='gildong', last_name='hong', 'DEF', param1='abcXYZ123987', age=27, 987, gender=True, param2=12345)
	args_kwargs_func(1, 3.7, 'abc', 'DEF', 987, first_name='gildong', last_name='hong', param1='abcXYZ123987', age=27, gender=True, param2=12345)

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
