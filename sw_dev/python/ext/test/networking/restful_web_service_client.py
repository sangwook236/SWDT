#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3.6/howto/urllib2.html

# NOTE [info] >> Use django REST framework as a RESTful web service.
#	http://www.django-rest-framework.org/tutorial/1-serialization/
#	http://www.django-rest-framework.org/tutorial/2-requests-and-responses/
#	http://www.django-rest-framework.org/tutorial/3-class-based-views/

import urllib.request
import urllib.parse
import urllib.error
import json, base64
import requests

# List all code snippets (GET).
def list_snippets(url):
	try:
		with urllib.request.urlopen(url) as response:
			#html = response.read()
			html = response.read().decode('utf-8')
			print(html)
			print(response.status, response.msg)
	except urllib.error.HTTPError as ex:
		print('HTTPError:', ex)

# Create a new snippet (POST).
def create_snippet(url):
	# Send files.
	#file_obj = open('data.txt', 'r')
	##file_obj = open('data.conf', 'r')
	##file_obj = open('data.josn', 'r')
	#file_obj = base64.b64encode(file_obj.read().encode('ascii'))

	data = {
		'title': 'test1',
		'code': 'System.io.println("Java");',
		'linenos': False,
		#'language': 'java',
		#'style': 'friendly',
		#'file': file_obj,
	}

	params = urllib.parse.urlencode(data)
	params = params.encode('ascii')  # Params should be bytes.
	request = urllib.request.Request(url, params)

	try:
		with urllib.request.urlopen(request) as response:
			#html = response.read()
			html = response.read().decode('utf-8')
			print(html)
			print(response.status, response.msg)
	except urllib.error.HTTPError as ex:
		print('HTTPError:', ex)

# Retrieve a code snippet (GET).
def retrieve_snippet(url, id):
	url = url + str(id) + '/'

	try:
		with urllib.request.urlopen(url) as response:
			#html = response.read()
			html = response.read().decode('utf-8')
			print(html)
			print(response.status, response.msg)
	except urllib.error.HTTPError as ex:
		print('HTTPError:', ex)

# Update a code snippet (PUT).
def update_snippet(url, id):
	url = url + str(id) + '/'

	data = {
		'title': 'ABC',
		'code': 'print(ABC)',
		'linenos': False,
		'language': 'java',
		'style': 'colorful',
	}

	params = urllib.parse.urlencode(data)
	params = params.encode('ascii')  # Params should be bytes.
	request = urllib.request.Request(url, params)
	request.get_method = lambda: 'PUT'

	try:
		with urllib.request.urlopen(request) as response:
			#html = response.read()
			html = response.read().decode('utf-8')
			print(html)
			print(response.status, response.msg)
	except urllib.error.HTTPError as ex:
		print('HTTPError:', ex)

# Delete a code snippet (DELETE).
def delete_snippet(url, id):
	url = url + str(id) + '/'

	request = urllib.request.Request(url)
	request.get_method = lambda: 'DELETE'

	try:
		with urllib.request.urlopen(request) as response:
			#html = response.read()
			html = response.read().decode('utf-8')
			#print(html)
			print(response.status, response.msg)
	except urllib.error.HTTPError as ex:
		print('HTTPError:', ex)

# Upload files (POST).
def upload_file(url):
	url = url + 'files/'

	# Send files.
	filename = 'data.txt'
	conf_filename = 'data.conf'
	json_filename = 'data.json'

	data = {
	}
	files = {
		'file': (filename, open(filename, 'rb'), 'multipart/form-data'),
		'conf_file': (conf_filename, open(conf_filename, 'rb'), 'multipart/form-data'),
		'json_file': (json_filename, open(json_filename, 'rb'), 'multipart/form-data'),
	}

	try:
		with requests.post(url, files=files, data=data) as response:
			print(response.text)
	except requests.HTTPError as ex:
		print('requests.HTTPError:', ex)
	except requests.RequestException as ex:
		print('requests.RequestException:', ex)

def main():
	#url = 'http://127.0.0.1:8000/snippets/'
	url = 'http://192.168.0.45:8001/snippets/'

	#list_snippets(url)
	#create_snippet(url)

	retrieve_snippet(url, 2)
	#update_snippet(url, 4)
	#delete_snippet(url, 51)

	#upload_file(url)

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
