#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3.6/library/socketserver.html

import socket
import sys

def main():
	HOST, PORT = 'localhost', 9999

	#data = ' '.join(sys.argv[1:])
	data = 'Hello, World!'

	# Create a socket (SOCK_STREAM means a TCP socket).
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
		# Connect to server and send data.
		sock.connect((HOST, PORT))
		sock.sendall(bytes(data + '\n', 'utf-8'))

		# Receive data from the server and shut down.
		received = str(sock.recv(1024), 'utf-8')

	print('Sent:     {}'.format(data))
	print('Received: {}'.format(received))

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
