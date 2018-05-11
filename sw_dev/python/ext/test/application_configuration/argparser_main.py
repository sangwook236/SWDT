#!/usr/bin/env python

import argparse

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('id', help="Id")
	parser.add_argument('-i', '--input-file', help='Input file')
	parser.add_argument('-c', '--config', default=None, help='Configuration file')
	parser.add_argument('-s', '--season', default='Spring', choices={'Spring', 'Summer', 'Fall', 'Winter'}, help='Season')

	args = parser.parse_args()

	print('Id = {}'.format(args.id))
	print('Input file = {}'.format(args.input_file))
	print('Config = {}'.format(args.config))
	print('Season = {}'.format(args.season))

#%%------------------------------------------------------------------

# Usage:
#	python argparser_main.py my-id -i input.dat -c config.json -s Winter
#	python argparser_main.py my-id --input-file=input.dat --config=config.json --season=Winter

if '__main__' == __name__:
	main()
