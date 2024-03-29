#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import glob, csv

def basic():
	filepath = './data.txt'
	try:
		with open(filepath, 'r', encoding='UTF8') as fd:
		#with open(filepath, 'r+') as fd:
		#with open(filepath, 'rb') as fd:
			data = fd.read()

		words = data.split()
	except UnicodeDecodeError as ex:
		print('Unicode decode error in {}: {}.'.format(filepath, ex))
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(filepath, ex))

	filepath = './data_copyed.txt'
	try:
		with open(filepath, 'w', encoding='UTF8') as fd:
		#with open(filepath, 'w+', encoding='UTF8') as fd:
		#with open(filepath, 'wb', encoding='UTF8') as fd:
			fd.write(data)
	except UnicodeDecodeError as ex:
		print('Unicode decode error in {}: {}.'.format(filepath, ex))
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(filepath, ex))

	filepath = './data_copyed.txt'
	try:
		with open(filepath, 'a', encoding='UTF8') as fd:
			fd.write(data)
	except UnicodeDecodeError as ex:
		print('Unicode decode error in {}: {}.'.format(filepath, ex))
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(filepath, ex))

	filepath = './data.txt'
	try:
		with open(filepath, 'r', encoding='UTF8') as fd:
			#lines = fd.read()  # A string.
			#lines = fd.read().strip('\n')  # A string.
			#lines = fd.read().replace(' ', '')  # A string.
			#lines = fd.readlines()  # A list of strings.
			lines = fd.read().splitlines()  # A list of strings.
	except UnicodeDecodeError as ex:
		print('Unicode decode error in {}: {}.'.format(filepath, ex))
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(filepath, ex))

	lines2 = list()
	for line in lines:
		#line = line.strip('\n').split(' ')
		line = line.rstrip('\n')
		lines2.append(line)

# Unix style pathname pattern expansion.
# REF [site] >> https://docs.python.org/3/library/glob.html
def glob_example():
	filepaths = glob.glob('./[0-9].*')
	filepaths = glob.glob('*.gif')
	filepaths = glob.glob('?.gif')
	filepaths = glob.glob('**/*.txt', recursive=True)
	filepaths = glob.glob('./**/', recursive=True)

# REF [site] >> https://docs.python.org/3/library/csv.html
def csv_example():
	if False:
		# For a CSV file.
		filepath = './test.csv'
		delimiter = ','
	else:
		# For a TSV file.
		filepath = './test.tsv'
		delimiter = '\t'
	quoting = csv.QUOTE_MINIMAL  # {csv.QUOTE_ALL, csv.QUOTE_NONNUMERIC, csv.QUOTE_MINIMAL}.
	quotechar = '"'
	escapechar = '\\'

	try:
		#with open(filepath, 'w', encoding='UTF8') as fd:
		with open(filepath, 'w', newline='', encoding='UTF8') as fd:
			writer = csv.writer(fd, delimiter=delimiter, quotechar=quotechar, quoting=quoting)
			#writer = csv.writer(fd, delimiter=delimiter, quotechar=quotechar, doublequote=False, escapechar=escapechar, quoting=quoting)
			#writer = csv.writer(fd, delimiter=delimiter, escapechar=escapechar, quoting=csv.QUOTE_NONE)
			#writer = csv.writer(fd, delimiter=delimiter, quotechar=None, escapechar=escapechar, quoting=csv.QUOTE_NONE)

			#writer.writerow(['id', 'lowercase1', 'uppercase1', 'lowercase2', 'uppercase2', 'class'])  # Writes a header.

			writer.writerow([1, 'a', 'A', 'aa,"bb', 'AA,BB', 0])
			writer.writerow([2, 'b', 'B', 'bb,cc', 'BB,CC', 1])
			writer.writerow([3, 'c', 'C', 'cc,dd', 'CC,DD', 0])
			writer.writerow([4, 'd', 'D', 'dd,ee', 'DD,EE', 1])
			writer.writerow([5, 'e', 'E', 'ee,ff', 'EE,FF', 0])
	except csv.Error as ex:
		print('csv.Error in {}: {}.'.format(filepath, ex))
	except UnicodeDecodeError as ex:
		print('Unicode decode error in {}: {}.'.format(filepath, ex))
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(filepath, ex))

	try:
		#with open(filepath, 'r', encoding='UTF8') as fd:
		with open(filepath, 'r', newline='', encoding='UTF8') as fd:
			has_header = csv.Sniffer().has_header(fd.read(1024))
			fd.seek(0)  # Rewind.

			reader = csv.reader(fd, delimiter=delimiter, quotechar=quotechar, quoting=quoting)
			#reader = csv.reader(fd, delimiter=delimiter, quotechar=quotechar, doublequote=False, escapechar=escapechar, quoting=quoting)
			#reader = csv.reader(fd, delimiter=delimiter, escapechar=escapechar, quoting=csv.QUOTE_NONE)
			#reader = csv.reader(fd, delimiter=delimiter, quotechar=None, escapechar=escapechar, quoting=csv.QUOTE_NONE)
			if has_header:
				header = next(reader, None)  # Reads a header.
				print('Header = {}.'.format(header))
			else:
				print('No header.')
			for idx, row in enumerate(reader):
				print('Row {}: {}.'.format(idx, row))
	except csv.Error as ex:
		print('csv.Error in {}: {}.'.format(filepath, ex))
	except UnicodeDecodeError as ex:
		print('Unicode decode error in {}: {}.'.format(filepath, ex))
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(filepath, ex))

def main():
	#basic()

	#glob_example()
	csv_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
