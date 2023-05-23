#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import configparser

# REF [site] >> https://docs.python.org/3/library/configparser.html
def simple_example():
	config = configparser.ConfigParser()

	"""
	[DEFAULT]
	ServerAliveInterval = 45
	Compression = yes
	CompressionLevel = 9
	ForwardX11 = yes

	[forge.example]
	User = hg

	[topsecret.server.example]
	Port = 50022
	ForwardX11 = no
	"""

	config['DEFAULT'] = {
		'ServerAliveInterval': '45',
		'Compression': 'yes',
		'CompressionLevel': '9',
	}

	config['forge.example'] = {}
	config['forge.example']['User'] = 'hg'

	config['topsecret.server.example'] = {}
	topsecret = config['topsecret.server.example']
	topsecret['Port'] = '50022'  # mutates the parser.
	topsecret['ForwardX11'] = 'no'  # same here.

	config['DEFAULT']['ForwardX11'] = 'yes'

	with open('./example.ini', 'w') as configfile:
		config.write(configfile)

	#-----
	print('-----')

	# NOTE [info] >>
	#	The only bit of magic involves the DEFAULT section which provides default values for all other sections.
	#	Note also that keys in sections are case-insensitive and stored in lowercase.

	config = configparser.ConfigParser()
	print(f'{config.sections()=}')

	successfully_read_files = config.read('./example.ini')  # Return list of successfully read files.
	print(f'{successfully_read_files=}')
	print(f'{config.sections()=}')

	print(f"{'forge.example' in config=}")
	print(f"{'python.org' in config=}")

	print(f"{config['forge.example']['User']=}")
	#print(f"{config['forge.example']['User1234567']=}")  # KeyError: 'User1234567'.
	print(f"{config['DEFAULT']['Compression']=}")

	topsecret = config['topsecret.server.example']
	print(f"{topsecret['ForwardX11']=}")
	print(f"{topsecret['Port']=}")

	for key in config['forge.example']:
		print(key)

	print(f"{config['forge.example']['ForwardX11']=}")

	#-----
	print('-----')

	# NOTE [info] >>
	#	It is possible to read several configurations into a single ConfigParser, where the most recently added configuration has the highest priority.
	#	Any conflicting keys are taken from the more recent configuration while the previously existing keys are retained.

	another_config = configparser.ConfigParser()
	successfully_read_files = another_config.read('./example.ini')  # Return list of successfully read files.
	print(f'{successfully_read_files=}')

	print(f"{another_config['topsecret.server.example']['Port']=}")

	another_config.read_string("[topsecret.server.example]\nPort=48484")
	print(f"{another_config['topsecret.server.example']['Port']=}")

	another_config.read_dict({"topsecret.server.example": {"Port": 21212}})
	print(f"{another_config['topsecret.server.example']['Port']=}")

	print(f"{another_config['topsecret.server.example']['ForwardX11']=}")

def main():
	simple_example()

#---------------------------------------------------------------------

if '__main__' == __name__:
	main()
