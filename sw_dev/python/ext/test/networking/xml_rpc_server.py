#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3.5/library/xmlrpc.server.html

from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
import datetime

class MyFuncs:
	def mul(self, x, y):
		return x * y

	@staticmethod
	def getCurrentTime():
		return datetime.datetime.now()

class ExampleService:
	def getData(self):
		return '42'

	class currentTime:
		@staticmethod
		def getCurrentTime():
			return datetime.datetime.now()

# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

def main():
	server = SimpleXMLRPCServer(('localhost', 8000), requestHandler=RequestHandler)
	server.register_introspection_functions()

	# Register pow() function: this will use the value of pow.__name__ as the name, which is just 'pow'.
	server.register_function(pow)

	# Register a function under a different name.
	server.register_function(lambda x, y: x + y, 'add')

	# NOTE [info] {important} >> Ths last instance is only registered.

	# Register an instance: all the methods of the instance are published as XML-RPC methods (in this case, just 'mul').
	server.register_instance(MyFuncs())

	# NOTE [info] {warning} >>
	#	Enabling the allow_dotted_names option allows intruders to access your module's global variables and may allow intruders to execute arbitrary code on your machine.
	#	Only use this example only within a secure, closed network.
	#server.register_instance(ExampleService(), allow_dotted_names=True)

	server.register_multicall_functions()

	print('Serving XML-RPC on localhost port 8000')

	try:
	    server.serve_forever()
	except KeyboardInterrupt:
	    print('\nKeyboard interrupt received, exiting.')
	    server.server_close()
	    sys.exit(0)

#%%------------------------------------------------------------------

# Usage:
#	python -m xml_rpc_server

if '__main__' == __name__:
	main()
