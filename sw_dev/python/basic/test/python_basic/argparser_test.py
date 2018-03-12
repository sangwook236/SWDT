import argparse

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('id', help="Id")
	parser.add_argument('-i', '--input-file', help="Input file")
	parser.add_argument('-c', '--config', default=None, help="Configuration file")

	args = parser.parse_args()
	
	print('Id = {}'.format(args.id))
	print('Input file = {}'.format(args.input_file))
	print('Config = {}'.format(args.config))

# Usage.
#	python argparser_test.py my-id -i input.dat -c config.json

if __name__ == '__main__':
	main()
