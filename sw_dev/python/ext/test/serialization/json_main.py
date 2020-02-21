#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json

def main():
	# File.
	with open('test.json', encoding='UTF8') as fd:
		json_data = json.load(fd)
		print(json_data)

	with open('tmp.json', 'w+', encoding='UTF8') as fd:
		json.dump(json_data, fd, indent='\t')

	# String.
	str = json.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}])
	print(type(str), str)

	lst = json.loads('["foo", {"bar":["baz", null, 1.0, 2]}]')
	print(type(lst), lst)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
