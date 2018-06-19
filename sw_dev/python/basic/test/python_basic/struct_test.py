#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3/library/struct.html
#	struct module performs conversions between Python values and C structs represented as Python bytes objects.

import struct
import sys

def main():
	#--------------------
	packet = struct.pack('hhl', 1, 2, 3)  # bytes.
	print('packet =', packet)

	packet1 = struct.unpack('hhl', packet)
	print('unpacked packet =', packet1)  # tuple: (1, 2, 3).

	#--------------------
	# Endian.

	print('byte order = {} endian.'.format(sys.byteorder))

	packet = struct.pack('hhl', 1, 2, 3)
	print('native =', packet)
	packet = struct.pack('<hhl', 1, 2, 3)  # Little endian.
	print('little-endian =', packet)
	packet = struct.pack('>hhl', 1, 2, 3)  # Big endian.
	print('big-endian =', packet)

	#--------------------
	record = b'raymond   \x32\x12\x08\x01\x08'
	name, serialnum, school, gradelevel = struct.unpack('<10sHHb', record)
	print(name, serialnum, school, gradelevel)

	# The ordering of format characters may have an impact on size since the padding needed to satisfy alignment requirements is different.
	pack1 = struct.pack('ci', b'*', 0x12131415)
	print(struct.calcsize('ci'), pack1)
	pack2 = struct.pack('ic', 0x12131415, b'*')
	print(struct.calcsize('ic'), pack2)

	# The following format 'llh0l' specifies two pad bytes at the end, assuming longs are aligned on 4-byte boundaries.
	print(struct.pack('llh0l', 1, 2, 3))

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
