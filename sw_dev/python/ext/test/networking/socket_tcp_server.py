#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3.6/library/socketserver.html

import socketserver

class MyTCPHandler(socketserver.BaseRequestHandler):
	def handle(self):
		# self.request is the TCP socket connected to the client.
		self.data = self.request.recv(1024).strip()
		print('{} wrote:'.format(self.client_address[0]))
		print(self.data)
		# Just send back the same data, but upper-cased.
		self.request.sendall(self.data.upper())

def main():
	HOST, PORT = 'localhost', 9999

	print('Listening on {}:{}'.format(HOST, PORT))

	# Create the server, binding to localhost on port 9999.
	with socketserver.TCPServer((HOST, PORT), MyTCPHandler) as server:
		# Activate the server.
		#	this will keep running until you interrupt the program with Ctrl-C.
		server.serve_forever()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
