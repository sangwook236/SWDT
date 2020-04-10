#!/usr/bin/env python

# REF [site] >> https://docs.python.org/3/library/struct.html
#	struct module performs conversions between Python values and C structs represented as Python bytes objects.

import struct
import sys

# REF [function] >> bytes_test() in basic.py
def basic_operation():
	#--------------------
	packet = struct.pack('hhl', 1, 2, 3)  # bytes.
	print('Packet =', packet)

	packet1 = struct.unpack('hhl', packet)
	print('Unpacked packet =', packet1)  # tuple: (1, 2, 3).

	#--------------------
	# Endian.

	print('Byte order = {} endian.'.format(sys.byteorder))

	packet = struct.pack('hhl', 1, 2, 3)
	print('Native        =', packet)
	packet = struct.pack('<hhl', 1, 2, 3)  # Little endian.
	print('Little-endian =', packet)
	packet = struct.pack('>hhl', 1, 2, 3)  # Big endian.
	print('Big-endian    =', packet)

	# NOTE [info] >> Native, 
	packet = struct.pack('BLLH', 1, 2, 3, 4)
	print('Native        =', packet)
	packet = struct.pack('<BLLH', 1, 2, 3, 4)  # Little endian.
	print('Little-endian =', packet)
	packet = struct.pack('>BLLH', 1, 2, 3, 4)  # Big endian.
	print('Big-endian    =', packet)

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

def IEEE_754_format()
	# IEEE 754 (binary64) <--> double precision floating-point number.
	ieee754_hex_strs = [
		'3FF0000000000000',  # 1.0.
		'4000000000000000',  # 2.0.
		'C000000000000000',  # -2.0.
		'4008000000000000',  # 3.0.
		'4010000000000000',  # 4.0.
		'4014000000000000',  # 5.0.
		'4018000000000000',  # 6.0.
		'4037000000000000',  # 23.0.
		'3F88000000000000',  # 0.01171875 = 3 / 256.
	]

	for hs in ieee754_hex_strs:
		dbl_val = struct.unpack('<d', struct.pack('<Q', int(hs, 16)))[0]
		hex_val = struct.unpack('<Q', struct.pack('<d', dbl_val))[0]

		print("'{}' (IEEE 754) -> {} (double) -> '{}' (IEEE 754).".format(hs, dbl_val, hex(hex_val)))

def main():
	#basic_operation()

	IEEE_754_format()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
