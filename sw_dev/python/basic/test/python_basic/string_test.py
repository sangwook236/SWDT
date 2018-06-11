#!/usr/bin/env python

def main():
	with open('data.txt', 'r') as myfile:
		data = myfile.read()

	words = data.split()

	with open('data.txt', 'r') as myfile:
		lines = myfile.readlines()

	lines2 = []
	for line in lines:
		lines2.append(line.rstrip('\n'))

	idx1 = data.find('Image loaded:')
	idx2 = data.rfind('Image loaded:')

	import re
	indices = [ss.start() for ss in re.finditer('Image loaded:', data)]

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
