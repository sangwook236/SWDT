#!/usr/bin/env python

import xml.etree.ElementTree as ET
import xmltodict

# REF [site] >> https://docs.python.org/3/library/xml.etree.elementtree.html
def simple_element_tree_example():
	country_data_as_string = """<?xml version="1.0"?>
		<data>
			<country name="Liechtenstein">
				<rank>1</rank>
				<year>2008</year>
				<gdppc>141100</gdppc>
				<neighbor name="Austria" direction="E"/>
				<neighbor name="Switzerland" direction="W"/>
			</country>
			<country name="Singapore">
				<rank>4</rank>
				<year>2011</year>
				<gdppc>59900</gdppc>
				<neighbor name="Malaysia" direction="N"/>
			</country>
			<country name="Panama">
				<rank>68</rank>
				<year>2011</year>
				<gdppc>13600</gdppc>
				<neighbor name="Costa Rica" direction="W"/>
				<neighbor name="Colombia" direction="E"/>
		    </country>
		</data>
		"""
	root_str = ET.fromstring(country_data_as_string)

	tree = ET.parse('country_data.xml')
	root = tree.getroot()

	for child in root:
		print(child.tag, child.attrib)
	print(root[0][1].text)

	for elem in root.findall('country'):
		print(elem.get('name'))

	print([elem.tag for elem in root.iter()])
	print(ET.tostring(root, encoding='utf8').decode('utf8'))

# REF [site] >> https://docs.python-guide.org/scenarios/xml/
def simple_xmltodict_example():
	filepath = 'country_data.xml'
	try:
		with open(filepath, 'r', encoding='UTF8') as fd:
			doc = xmltodict.parse(fd.read())
	except FileNotFoundError as ex:
		print('File not found: {}.'.format(filepath))
		raise
	except UnicodeDecodeError as ex:
		print('Unicode decode error: {}.'.format(filepath))
		raise

	print(doc['data']['country'])
	print(doc['data']['country'][0]['@name'])
	print(doc['data']['country'][2]['rank'])
	print(doc['data']['country'][2]['neighbor'][1]['@direction'])

def main():
	simple_element_tree_example()
	simple_xmltodict_example()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
