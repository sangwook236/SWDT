#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import yaml

class Hero:
	def __init__(self, name, hp, sp):
		self.name = name
		self.hp = hp
		self.sp = sp

	def __repr__(self):
		return "%s(name=%r, hp=%r, sp=%r)" % (self.__class__.__name__, self.name, self.hp, self.sp)

# REF [site] >> https://pyyaml.org/wiki/PyYAMLDocumentation
def simple_example():
	document = """
  a: 1
  b:
    c: 3
    d: 4
"""
	data = yaml.dump(yaml.load(document))
	#data = yaml.dump(yaml.load(document), default_flow_style=False)
	print(data)

	documents = """
---
name: The Set of Gauntlets 'Pauraegen'
description: >
    A set of handgear with sparks that crackle
    across its knuckleguards.
---
name: The Set of Gauntlets 'Paurnen'
description: |
  A set of gauntlets that gives off a foul,
  acrid odour yet remains untarnished.
---
name: The Set of Gauntlets 'Paurnimmen'
description: >
  A set of handgear, freezing with unnatural cold.
"""
	for data in yaml.load_all(documents):
		print(data)

	ss = yaml.dump({'name': 'Silenthand Olleander', 'race': 'Human', 'traits': ['ONE_HAND', 'ONE_EYE']})  # str.
	#ss = yaml.dump(range(50))
	#ss = yaml.dump(range(50), width=50, indent=4)
	#ss = yaml.dump(range(5), canonical=True)
	#ss = yaml.dump(range(5), default_flow_style=False)
	#ss = yaml.dump(range(5), default_flow_style=True, default_style='"')
	#ss = yaml.dump({'hyundai': 45000, 'tesla': 65000, 'chevrolet': 42000, 'audi': 51000, 'mercedesbenz': 80000}, sort_keys=True)
	print(ss)

	#--------------------
	# Construct a Python object of any type.
	data = yaml.load("""
none: [~, null]
bool: [true, false, on, off]
int: 42
float: 3.14159
list: [LITE, RES_ACID, SUS_DEXT]
dict: {hp: 13, sp: 5}
""", Loader=yaml.Loader)
	print(data)

	data = yaml.load("""
!!python/object:__main__.Hero
name: Welthyr Syxgon
hp: 1200
sp: 0
""", Loader=yaml.Loader)
	print(data)

	# Python objects to be serialized into a YAML document.
	data = yaml.dump([1, 2, 3], explicit_start=True)
	print(data)

	data = yaml.dump_all([1, 2, 3], explicit_start=True)
	print(data)

	data = yaml.dump(Hero("Galain Ysseleg", hp=-3, sp=2))
	print(data)

	#--------------------
	# REF [site] >> https://rfriend.tistory.com/540
	try:
		filepath = "./vegetables.yml"
		with open(filepath, encoding='utf-8') as fd:
			data = yaml.load(fd, Loader=yaml.FullLoader)
			print(data)
	except yaml.scanner.ScannerError as ex:
		print('yaml.scanner.ScannerError in {}: {}.'.format(filepath, ex))
	except UnicodeDecodeError as ex:
		print('Unicode decode error in {}: {}.'.format(filepath, ex))
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(filepath, ex))

	try:
		filepath = './tmp.yml'
		with open(filepath, 'w', encoding='utf-8') as fd:
			yaml.dump(data, fd)
			#yaml.dump(data, fd, indent=4)
	except UnicodeDecodeError as ex:
		print('Unicode decode error in {}: {}.'.format(filepath, ex))
	except FileNotFoundError as ex:
		print('File not found, {}: {}.'.format(filepath, ex))

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()