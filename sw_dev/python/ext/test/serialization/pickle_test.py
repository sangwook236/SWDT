#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pickle

def main():
	filepath = 'test.pkl'

	try:
		favorite_color = {'lion': 'yellow', 'kitty': 'red'}
		with open(filepath, 'wb') as fd:
			pickle.dump(favorite_color, fd)
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(filepath))

	try:
		with open(filepath, 'rb') as fd:
			loaded = pickle.load(fd)
			print(loaded)
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(filepath))

	#--------------------
	dumped = pickle.dumps(favorite_color)  # A bytes object.
	print(type(dumped))
	dummy = pickle.loads(dumped)
	print(dummy)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
