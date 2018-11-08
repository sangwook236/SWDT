#!/usr/bin/env python

def main():
	with open('data.txt', 'r') as fd:
	#with open('data.txt', 'r+') as fd:
	#with open('data.txt', 'rb') as fd:
		data = fd.read()

	with open('data_copyed.txt', 'w') as fd:
	#with open('data_copyed.txt', 'w+') as fd:
	#with open('data_copyed.txt', 'wb') as fd:
		fd.write(data)

	words = data.split()

	with open('data.txt', 'r') as fd:
		lines = fd.readlines()

	lines2 = []
	for line in lines:
		lines2.append(line.rstrip('\n'))

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
