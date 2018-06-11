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

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
