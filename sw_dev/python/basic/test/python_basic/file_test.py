#!/usr/bin/env python

import glob, csv

def basic():
	try:
		try:
			with open('data.txt', 'r', encoding='UTF8') as fd:
			#with open('data.txt', 'r+') as fd:
			#with open('data.txt', 'rb') as fd:
				data = fd.read()
		except FileNotFoundError as ex:
			print('File not found: {}.'.format('data.txt'))

		with open('data_copyed.txt', 'w', encoding='UTF8') as fd:
		#with open('data_copyed.txt', 'w+', encoding='UTF8') as fd:
		#with open('data_copyed.txt', 'wb', encoding='UTF8') as fd:
			fd.write(data)

		words = data.split()

		try:
			with open('data.txt', 'r') as fd:
				lines = fd.readlines()
		except FileNotFoundError as ex:
			print('File not found: {}.'.format('data.txt'))

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
	filepaths = glob.glob('./[0-9].*')
	filepaths = glob.glob('*.gif')
	filepaths = glob.glob('?.gif')
	filepaths = glob.glob('**/*.txt', recursive=True)
	filepaths = glob.glob('./**/', recursive=True)

def csv_example():
	csv_filepath = './test.csv'

	#with open(csv_filepath, 'w', encoding='UTF8') as csvfile:
	with open(csv_filepath, 'w', newline='', encoding='UTF8') as csvfile:
		writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

		#writer.writerow(['id', 'lowercase1', 'uppercase1', 'lowercase2', 'uppercase2', 'class'])  # Writes a header.

		writer.writerow([1, 'a', 'A', 'aa,bb', 'AA,BB', 0])
		writer.writerow([2, 'b', 'B', 'bb,cc', 'BB,CC', 1])
		writer.writerow([3, 'c', 'C', 'cc,dd', 'CC,DD', 0])
		writer.writerow([4, 'd', 'D', 'dd,ee', 'DD,EE', 1])
		writer.writerow([5, 'e', 'E', 'ee,ff', 'EE,FF', 0])

	#with open(csv_filepath, 'r', encoding='UTF8') as fd:
	with open(csv_filepath, 'r', newline='', encoding='UTF8') as fd:
		has_header = csv.Sniffer().has_header(fd.read(1024))
		fd.seek(0)  # Rewind.

		reader = csv.reader(fd, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		if has_header:
			header = next(reader, None)  # Reads a header.
			print('Header = {}.'.format(header))
		else:
			print('No header.')
		for idx, row in enumerate(reader):
			print('Row {}: {}.'.format(idx, row))

def main():
	#basic()

	#glob_example()
	csv_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
