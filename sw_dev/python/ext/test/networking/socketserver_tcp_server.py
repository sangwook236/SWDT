#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3.6/library/socketserver.html

import socketserver
import threading

class EchoBaseRequestHandler(socketserver.BaseRequestHandler):
	def setup(self):
		print('[{}] {} connected.'.format(threading.get_ident(), self.client_address))
		self.request.send(bytes('Hi ' + str(self.client_address) + '\n', 'utf-8'))

	def finish(self):
		self.request.send(bytes('Bye ' + str(self.client_address) + '\n', 'utf-8'))
		print('[{}] {} disconnected.'.format(threading.get_ident(), self.client_address))

	def handle(self):
		print('[{}] Enter EchoBaseRequestHandler.handle()'.format(threading.get_ident()))
		while True:
			recv_data = self.request.recv(1024)
			if len(recv_data) > 0:
				print('[{}] {} wrote: {}'.format(threading.get_ident(), self.client_address[0], recv_data))

				#self.request.send(recv_data)
				self.request.sendall(recv_data.upper())
				if 'bye' == str(recv_data.strip(), 'utf-8'):
					break
		print('[{}] Exit EchoBaseRequestHandler.handle()'.format(threading.get_ident()))

	def handle_old(self):
		# self.request is the TCP socket connected to the client.
		recv_data = self.request.recv(1024)
		print('[{}] {} wrote: {}'.format(threading.get_ident(), self.client_address[0], recv_data))

		# Just send back the same recv_data, but upper-cased.
		self.request.sendall(recv_data.upper())

class EchoStreamRequestHandler(socketserver.StreamRequestHandler):
	def setup(self):
		print('[{}] {} connected.'.format(threading.get_ident(), self.client_address))
		self.request.send(bytes('Hi ' + str(self.client_address) + '\n', 'utf-8'))

	def finish(self):
		self.request.send(bytes('Bye ' + str(self.client_address) + '\n', 'utf-8'))
		print('[{}] {} disconnected.'.format(threading.get_ident(), self.client_address))

	def handle(self):
		# self.rfile is a file-like object created by the handler:
		# we can now use e.g. readline() instead of raw recv() calls.
		recv_data = self.rfile.readline().strip()
		print('[{}] {} wrote: {}'.format(threading.get_ident(), self.client_address[0], recv_data))

		# Likewise, self.wfile is a file-like object used to write back to the client.
		self.wfile.write(recv_data.upper())

class MyTCPServer(socketserver.TCPServer):
	import socket

	address_family = socket.AF_INET
	allow_reuse_address = True

	def server_bind(self):
		self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.socket.bind(self.server_address)

# REF [site] >> https://www.programcreek.com/python/example/73643/SocketServer.BaseRequestHandler
class MyThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
	pass

def main():
	HOST, PORT = 'localhost', 9999
	#HOST, PORT = '192.168.10.2', 6789

	print('[{}] Listening on {}:{}'.format(threading.get_ident(), HOST, PORT))

	# Create the server, binding to HOST on port PORT.
	#with socketserver.TCPServer((HOST, PORT), EchoBaseRequestHandler) as server:
	#with socketserver.TCPServer((HOST, PORT), EchoStreamRequestHandler) as server:
	with socketserver.ThreadingTCPServer((HOST, PORT), EchoBaseRequestHandler) as server:
		# Activate the server.
		#	This will keep running until you interrupt the program with Ctrl-C.
		try:
			server.serve_forever()
		except KeyboardInterrupt:
			pass
	server.server_close()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
