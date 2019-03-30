#!/usr/bin/env python

import glob

def basic():
	try:
		with open('data.txt', 'r', encoding='UTF8') as fd:
		#with open('data.txt', 'r+') as fd:
		#with open('data.txt', 'rb') as fd:
			data = fd.read()

		with open('data_copyed.txt', 'w', encoding='UTF8') as fd:
		#with open('data_copyed.txt', 'w+', encoding='UTF8') as fd:
		#with open('data_copyed.txt', 'wb', encoding='UTF8') as fd:
			fd.write(data)

		words = data.split()

		with open('data.txt', 'r') as fd:
			lines = fd.readlines()

		lines2 = list()
		for line in lines:
			#line = line.strip('\n').split(' ')
			line = line.rstrip('\n')
			lines2.append(line)
	except FileNotFoundError as ex:
		print('FileNotFoundError raised:', ex)

# Unix style pathname pattern expansion.
# REF [site] >> https://docs.python.org/3/library/glob.html
def glob_example():
	glob.glob('./[0-9].*')
	glob.glob('*.gif')
	glob.glob('?.gif')
	glob.glob('**/*.txt', recursive=True)
	glob.glob('./**/', recursive=True)

def main():
	basic()

	glob_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
