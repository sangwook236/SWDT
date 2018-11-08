#!/usr/bin/env python

import json

def main():
	# File.
	with open('test.json') as json_file:
		json_data = json.load(json_file)
		print(json_data)

	with open('tmp.json', 'w+') as json_file:
		json.dump(json_data, json_file, indent='\t')

	# String.
	str = json.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}])
	print(type(str), str)

	lst = json.loads('["foo", {"bar":["baz", null, 1.0, 2]}]')
	print(type(lst), lst)

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
