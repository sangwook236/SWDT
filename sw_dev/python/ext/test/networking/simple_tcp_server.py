#!/usr/bin/env python

# REF [site] >> https://gist.github.com/Integralist/3f004c3594bbf8431c15ed6db15809ae

import socket
import threading

def handle_client_connection(client_socket):
	request = client_socket.recv(1024)
	print('Received {}'.format(request))
	#client_socket.send(bytes('ACK!', 'utf-8'))
	#client_socket.close()

def main():
	HOST, PORT = 'localhost', 9999
	#HOST, PORT = '192.168.10.8', 6789

	server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server.bind((HOST, PORT))
	server.listen(5)  # Max backlog of connections.

	print('Listening on {}:{}'.format(HOST, PORT))

	while True:
		client_sock, address = server.accept()
		print('Accepted connection from {}:{}'.format(address[0], address[1]))
		client_handler = threading.Thread(
			target=handle_client_connection,
			args=(client_sock,)  # Without comma you'd get a... TypeError: handle_client_connection() argument after * must be a sequence, not _socketobject.
		)
		client_handler.start()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
