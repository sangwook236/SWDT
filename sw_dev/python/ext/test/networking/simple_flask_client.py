#!/usr/bin/env python

import urllib.request
import urllib.parse
import urllib.error
import requests
import json, base64
import os

# Lists (GET).
def list(url):
	try:
		with urllib.request.urlopen(url) as response:
			#html = response.read()
			html = response.read().decode('utf-8')
			#print(html)
			with open('list.html', 'w') as file:
				file.write(html)
			print(response.status, response.msg)
	except urllib.error.HTTPError as ex:
		print('HTTPError:', ex)

# Create (POST).
def create(url):
	raise NotImplementedError

# Retrieve (GET).
def retrieve(url, id):
	raise NotImplementedError

# Update (PUT).
def update(url, id):
	raise NotImplementedError

# Delete (DELETE).
def delete(url, id):
	raise NotImplementedError

# Download a file (GET).
def download_file(url):
	try:
		with urllib.request.urlopen(url) as response:
			headers = response.getheaders()
			attachment = response.getheader('Content-Disposition')
			istart = attachment.find('filename=')
			istart = attachment.find('=', istart + 1) + 1
			filename = attachment[istart:]
			file_obj = response.read()
			if file_obj is not None:
				with open(os.path.join('./', filename + '.copyed'), 'wb') as file:
					file.write(file_obj)
			print('File downloaded.')
			print(response.status, response.msg)
	except urllib.error.HTTPError as ex:
		print('HTTPError:', ex)

# Upload files (POST).
def upload_file(url):
	txt_filename = 'data.txt'
	conf_filename = 'data.conf'
	json_filename = 'data.json'

	data = None
	files = {
		'file': (txt_filename, open(txt_filename, 'rb'), 'multipart/form-data'),
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
	url = 'http://localhost:8888/'

	#----------------------------------------------------------------
	#list(url + 'list/')
	#list(url + 'list_all/')

	#create_snippet(url + 'create/')
	#retrieve_snippet(url + 'retrieve/', 2)
	#update_snippet(url + 'update/', 4)
	#delete_snippet(url + 'delete/', 51)

	#----------------------------------------------------------------
	#download_file(url + 'download/data.txt')
	upload_file(url + 'upload/')

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
