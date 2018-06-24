#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3/library/http.server.html

import http.server
import threading

class MyBaseHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
	def do_POST(self):
		pass

class MySimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
	def do_POST(self):
		"""
		if self.path.startswith('/kill_server'):
			print('Server is going down, run it again manually!')
			def kill_me_please(server):
				server.shutdown()
			kill_worker = threading.Thread(target=kill_me_please, args=(httpd,))
			kill_worker.start()
			self.send_error(500)
		"""
		pass

# TODO [check] >> 
class MyHTTPServer(http.server.HTTPServer):
	import socket

	address_family = socket.AF_INET
	allow_reuse_address = True

	def server_bind(self):
		self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.socket.bind(self.server_address)

def main():
	HOST, PORT = 'localhost', 9999
	#HOST, PORT = '192.168.10.8', 6789

	print('Listening on {}:{}'.format(HOST, PORT))

	# Create the server, binding to HOST on port PORT.
	with http.server.HTTPServer((HOST, PORT), MyBaseHTTPRequestHandler) as server:
	#with http.server.HTTPServer((HOST, PORT), MySimpleHTTPRequestHandler) as server:
		# Activate the server.
		#	This will keep running until you interrupt the program with Ctrl-C.
		try:
			server.serve_forever()
		except KeyboardInterrupt:
			pass

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
