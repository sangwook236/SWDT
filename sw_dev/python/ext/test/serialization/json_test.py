#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json

def json_test():
	# File.
	try:
		filepath = './test.json'
		with open(filepath, encoding='utf-8') as fd:
			json_data = json.load(fd)
			print(json_data)
	except json.decoder.JSONDecodeError as ex:
		print(f'JSON decode error in {filepath}: {ex}.')
	except UnicodeDecodeError as ex:
		print(f'Unicode decode error in {filepath}: {ex}.')
	except FileNotFoundError as ex:
		print(f'File not found, {filepath}: {ex}.')

	try:
		filepath = './tmp.json'
		with open(filepath, 'w+', encoding='utf-8') as fd:
			json.dump(json_data, fd, indent='\t')
			#json.dump(json_data, fd, indent='    ')
	except UnicodeDecodeError as ex:
		print(f'Unicode decode error in {filepath}: {ex}.')
	except FileNotFoundError as ex:
		print(f'File not found, {filepath}: {ex}.')

	#--------------------
	# String.
	sz = json.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}])
	print(type(sz), sz)

	lst = json.loads('["foo", {"bar":["baz", null, 1.0, 2]}]')
	print(type(lst), lst)

def json_lines_test():
	raw_data = [
		{'name': 'Gilbert', 'wins': [['straight', '7♣'], ['one pair', '10♥']]},
		{'name': 'Alexa', 'wins': [['two pair', '4♠'], ['two pair', '9♠']]},
		{'name': 'May', 'wins': []},
		{'name': 'Deloise', 'wins': [['three of a kind', '5♣']]},
	]

	# Text file.
	filepath = './text.jsonl'
	try:
		with open(filepath, 'w', encoding='utf-8') as fd:
			for line in raw_data:
				fd.write(json.dumps(line) + '\n')
	except UnicodeDecodeError as ex:
		print(f'Unicode decode error in {filepath}: {ex}.')
	except FileNotFoundError as ex:
		print(f'File not found, {filepath}: {ex}.')

	try:
		with open(filepath, encoding='utf-8') as fd:
			jsonl_lines = fd.readlines()

		jsonl_lines = list(json.loads(line) for line in jsonl_lines)
		print(jsonl_lines)
	except json.decoder.JSONDecodeError as ex:
		print(f'JSON decode error in {filepath}: {ex}.')
	except UnicodeDecodeError as ex:
		print(f'Unicode decode error in {filepath}: {ex}.')
	except FileNotFoundError as ex:
		print(f'File not found, {filepath}: {ex}.')

	#--------------------
	# Binary file.
	filepath = './binary.jsonl'
	try:
		with open(filepath, 'wb') as fd:  # TODO [check] >> Binary file?
			for line in raw_data:
				fd.write((json.dumps(line) + '\n').encode('utf-8'))
	except UnicodeDecodeError as ex:
		print(f'Unicode decode error in {filepath}: {ex}.')
	except FileNotFoundError as ex:
		print(f'File not found, {filepath}: {ex}.')

	try:
		with open(filepath, 'rb') as fd:
			jsonl_lines = fd.readlines()

		jsonl_lines = list(json.loads(line.decode('utf-8').strip('\n')) for line in jsonl_lines)
		print(jsonl_lines)
	except json.decoder.JSONDecodeError as ex:
		print(f'JSON decode error in {filepath}: {ex}.')
	except UnicodeDecodeError as ex:
		print(f'Unicode decode error in {filepath}: {ex}.')
	except FileNotFoundError as ex:
		print(f'File not found, {filepath}: {ex}.')

def main():
	#json_test()
	json_lines_test()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
