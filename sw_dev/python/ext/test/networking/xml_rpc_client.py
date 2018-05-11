#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3.5/library/xmlrpc.server.html

import xmlrpc.client

def main():
	server = xmlrpc.client.ServerProxy('http://localhost:8000')

	try:
		print(server.pow(2, 9))
		print(server.add(1, 2))
		print(server.mul(5, 2))
		print(server.getCurrentTime())
		#print(server.getData())
		#print(server.currentTime.getCurrentTime())
	except Exception as ex:
		print('Exception:', ex)

	multi = xmlrpc.client.MultiCall(server)
	multi.pow(2, 9)
	multi.add(1, 2)
	multi.mul(5, 2)
	multi.getCurrentTime()
	#multi.getData()
	#multi.currentTime.getCurrentTime()

	try:
		for response in multi():
			print(response)
	except Exception as ex:
		print('Exception:', ex)

#%%------------------------------------------------------------------

# Usage:
#	python -m xml_rpc_client

if '__main__' == __name__:
	main()
