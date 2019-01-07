#!/usr/bin/env python

def main():
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

		lines2 = []
		for line in lines:
			lines2.append(line.rstrip('\n'))
	except FileNotFoundError as ex:
		print('FileNotFoundError raised:', ex)

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
