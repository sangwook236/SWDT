#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3.6/library/socketserver.html

import socket
import sys, time
import struct

def send_msg(sock, msg):
	#packet = struct.pack('>s', msg)
	packet = bytes(msg, 'utf-8')

	# For string msg.
	#packet = bytes(str, 'utf-8')
	#packet = struct.pack('I{}s'.format(len(packet)), len(packet), packet)
	#print('Sent: {}'.format(str))

	sock.sendall(packet)
	print('Sent: {}'.format(msg))

def receive_msg(sock):
	# Receive data from the server (and shut down).
	#recv_data = sock.recv(1024)
	#if len(recv_data) > 0:
	#	received = str(recv_data, 'utf-8')
	#	print('Received: {}'.format(received))

	data = b''
	packet = sock.recv(1024)
	if not packet:
		print('No recv data.')
		return None
	data += packet

	# For string msg.
	#(idx,), data = struct.unpack('I', data[:4]), data[4:]
	#str, data = data[:idx], data[idx:]
	#print('Received: {}, {}'.format(str, data))

	#data = struct.unpack('>s', data)
	data = str(data, 'utf-8')
	print('Received: {}'.format(data))

	return data

def main():
	HOST, PORT = 'localhost', 9999

	# Create a socket (SOCK_STREAM means a TCP socket).
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
		try:
			#--------------------
			# Connect to server.
			sock.connect((HOST, PORT))

			recv_data = receive_msg(sock)

			#--------------------
			#send_data = ' '.join(sys.argv[1:])
			#send_data = 'Hello, World!\n'
			send_data = 'Hello, World!'
			send_msg(sock, send_data)

			recv_data = receive_msg(sock)

			#--------------------
			#send_data = 'abcdef-1234567890\n'
			send_data = 'abcdef-1234567890'
			send_msg(sock, send_data)

			recv_data = receive_msg(sock)

			#--------------------
			send_data = 'bye'
			send_msg(sock, send_data)

			recv_data = receive_msg(sock)

			recv_data = receive_msg(sock)

			#time.sleep(1)
		except ConnectionRefusedError as ex:
			print('ConnectionRefusedError raised:', ex)
		except ConnectionAbortedError as ex:
			print('ConnectionAbortedError raised:', ex)
		except:
			print('Unexpected error raised:', sys.exc_info()[0])
		finally:
			sock.close()
			print('Close socket.')

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
