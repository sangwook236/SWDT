#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3.6/howto/urllib2.html

# NOTE [info] >> Use django REST framework as a RESTful web service.
#	http://www.django-rest-framework.org/tutorial/1-serialization/
#	http://www.django-rest-framework.org/tutorial/2-requests-and-responses/
#	http://www.django-rest-framework.org/tutorial/3-class-based-views/
# NOTE [info] >>
#	https://docs.djangoproject.com/en/2.1/topics/http/file-uploads/

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
	# Send files.
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

# List all jobs (GET).
def list_jobs(url):
	try:
		with urllib.request.urlopen(url) as response:
			#html = response.read()
			html = response.read().decode('utf-8')
			print(html)
			print(response.status, response.msg)
	except urllib.error.HTTPError as ex:
		print('HTTPError:', ex)

# Create a job (POST).
def create_job(url):
	# Job file.
	job_filename = 'data.json'

	data = {
		'job_name': 'job0001',
		'job_code': 'spark',
		'job_description': 'This is a test job.',
		'status': 'created',
		'exit_code': 'undefined',
	}
	files = {
		'job_file': (job_filename, open(job_filename, 'rb'), 'multipart/form-data'),
	}

	try:
		with requests.post(url, files=files, data=data) as response:
			print(response.text)
	except requests.HTTPError as ex:
		print('requests.HTTPError:', ex)
	except requests.RequestException as ex:
		print('requests.RequestException:', ex)

# Retrieve a job (GET).
def retrieve_job(url, id):
	url = url + str(id) + '/'

	try:
		with urllib.request.urlopen(url) as response:
			#html = response.read()
			#html = response.read().decode('utf-8')
			#print(html)
			headers = response.getheaders()
			attachment = response.getheader('Content-Disposition')
			istart = attachment.find('filename=')
			istart = attachment.find('"', istart + 1) + 1
			iend = attachment.find('"', istart + 1)
			filename = attachment[istart:iend]
			file_obj = response.read()
			if file_obj is not None:
				dst = open('./' + filename, 'wb')
				dst.write(file_obj)
				dst.close()
			print('File downloaded.')
			print(response.status, response.msg)
	except urllib.error.HTTPError as ex:
		print('HTTPError:', ex)

# Update a job (PUT).
def update_job(url, id):
	url = url + str(id) + '/'

	data = {
		'job_name': 'job1000',
		'job_code': 'hadoop',
		'job_description': 'This is an updated job.',
		'status': 'updated',
		'exit_code': 'succeeded',
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

# Delete a job (DELETE).
def delete_job(url, id):
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

def main():
	#url = 'http://127.0.0.1:8000/'
	url = 'http://192.168.0.45:8001/'

	#----------------------------------------------------------------
	#list_snippets(url + 'snippets/')
	#create_snippet(url + 'snippets/')

	#retrieve_snippet(url + 'snippets/', 2)
	#update_snippet(url + 'snippets/', 4)
	#delete_snippet(url + 'snippets/', 51)

	#----------------------------------------------------------------
	#upload_file(url + 'files/')

	#----------------------------------------------------------------
	#list_jobs(url + 'jobs/')
	#create_job(url + 'jobs/')

	retrieve_job(url + 'jobs/', 2)
	#update_job(url + 'jobs/', 4)
	#delete_job(url + 'jobs/', 6)

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
