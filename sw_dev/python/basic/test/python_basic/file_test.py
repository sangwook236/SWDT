#!/usr/bin/env python

def main():
	with open('data.txt', 'r') as file:
	#with open('data.txt', 'r+') as file:
	#with open('data.txt', 'rb') as file:
		data = file.read()

	with open('data_copyed.txt', 'w') as file:
	#with open('data_copyed.txt', 'w+') as file:
	#with open('data_copyed.txt', 'wb') as file:
		file.write(data)

	words = data.split()

	with open('data.txt', 'r') as myfile:
		lines = myfile.readlines()

	lines2 = []
	for line in lines:
		lines2.append(line.rstrip('\n'))

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
