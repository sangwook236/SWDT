#!/usr/bin/env python

# REF [site] >> https://gist.github.com/Integralist/3f004c3594bbf8431c15ed6db15809ae

import socket

def main():
	hostname, port = 'localhost', 6789

	# Create an ipv4 (AF_INET) socket object using the tcp protocol (SOCK_STREAM).
	client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

	# Connect the client.
	client.connect((hostname, port))

	# Send some data (in this case a HTTP GET request).
	client.send(bytes('GET /index.html HTTP/1.1\r\nHost: {}\r\n\r\n'.format(hostname), 'utf-8'))

	# Receive the response data (4096 is recommended buffer size).
	response = client.recv(4096)

	print(response)

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
