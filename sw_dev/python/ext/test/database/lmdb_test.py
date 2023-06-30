#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, json
import numpy as np
#os.environ['LMDB_FORCE_CFFI'] = '1'
import lmdb

def basic_operation():
	try:
		lmdb_dir_path = './mylmdb'  # Creates ./mylmdb/data.mdb & ./mylmdb/lock.mdb.
		with lmdb.open(lmdb_dir_path, map_size=1024**2, max_dbs=10, subdir=True, readonly=False, create=True) as env:
		#lmdb_file_prefix = './mylmdb/lmdb_file'  # Creates ./mylmdb/lmdb_file & ./mylmdb/lmdb_filelock.
		#with lmdb.open(lmdb_file_prefix, map_size=1024**2, max_dbs=10, subdir=False, readonly=False, create=True) as env:
			with env.begin(write=True) as txn:  # A transaction object
				for id in range(100):
					key, val = f'id_{id}', f'{id}'
					txn.put(key.encode('ascii'), val.encode('ascii'))
	except lmdb.MapFullError as ex:
		print(f'lmdb.MapFullError raised: {ex}.')

	try:
		lmdb_dir_path = './mylmdb'
		with lmdb.open(lmdb_dir_path, readonly=True) as env:
			with env.begin() as txn:  # A transaction object.
				cursor = txn.cursor()
				for key, value in cursor:
					#print(f'{key}: {value}.')
					print(f"{key.decode('ascii')}: {value.decode('ascii')}.")

				for id in range(100):
					key = f'id_{id}'
					val = txn.get(key.encode('ascii'))
					print(f"{key} = {val.decode('ascii')}.")
	except lmdb.MapFullError as ex:
		print(f'lmdb.MapFullError raised: {ex}.')

def write_to_db_example(use_caffe_datum=False):
	N = 1000
	X = np.zeros((N, 3, 32, 32), dtype=np.uint8)
	y = np.zeros(N, dtype=np.int64)

	try:
		lmdb_dir_path = './mylmdb'
		map_size = X.nbytes * 10
		with lmdb.open(lmdb_dir_path, map_size=map_size) as env:
			with env.begin(write=True) as txn:  # A transaction object.
				if use_caffe_datum:
					#from caffe.proto import caffe_pb2
					import caffe_pb2

					for i in range(N):
						# REF [site] >> https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto
						datum = caffe_pb2.Datum()
						datum.channels = X.shape[1]
						datum.height = X.shape[2]
						datum.width = X.shape[3]
						datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9.
						datum.label = int(y[i])
						str_id = '{:08}'.format(i)

						# The encode is only essential in Python 3.
						txn.put(str_id.encode('ascii'), datum.SerializeToString())
				else:
					for i in range(N):
						datum = {
							'channels': X.shape[1],
							'height': X.shape[2],
							'width': X.shape[3],
							'data': X[i].tolist(),
							'label': int(y[i]),
						}
						str_id = '{:08}'.format(i)

						# The encode is only essential in Python 3.
						txn.put(str_id.encode('ascii'), json.dumps(datum).encode('ascii'))

			#--------------------
			print(env.stat())
	except lmdb.MapFullError as ex:
		print('lmdb.MapFullError raised: {}.'.format(ex))

def read_from_db_example(use_caffe_datum=False):
	lmdb_dir_path = './mylmdb'
	with lmdb.open(lmdb_dir_path, readonly=True) as env:
		with env.begin() as txn:
			raw_datum = txn.get(b'00000000')

	if use_caffe_datum:
		#from caffe.proto import caffe_pb2
		import caffe_pb2

		# REF [site] >> https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto
		datum = caffe_pb2.Datum()
		datum.ParseFromString(raw_datum)

		x = np.fromstring(datum.data, dtype=np.uint8)
		x = x.reshape(datum.channels, datum.height, datum.width)
		y = datum.label
	else:
		datum = json.loads(raw_datum.decode('ascii'))

		x = np.array(datum['data'], dtype=np.uint8)
		x = x.reshape(datum['channels'], datum['height'], datum['width'])
		y = datum['label']

	print(x.shape, y)

def key_value_example(use_caffe_datum=False):
	lmdb_dir_path = './mylmdb'
	with lmdb.open(lmdb_dir_path, readonly=True) as env:
		with env.begin() as txn:
			cursor = txn.cursor()
			if use_caffe_datum:
				#from caffe.proto import caffe_pb2
				import caffe_pb2

				for k, v in cursor:
					# REF [site] >> https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto
					datum = caffe_pb2.Datum()
					datum.ParseFromString(v)

					x = np.fromstring(datum.data, dtype=np.uint8)
					x = x.reshape(datum.channels, datum.height, datum.width)
					y = datum.label
					print(k.decode(), x.shape, y)
			else:
				for k, v in cursor:
					datum = json.loads(v.decode('ascii'))
					x = np.array(datum['data'], dtype=np.uint8)
					x = x.reshape(datum['channels'], datum['height'], datum['width'])
					y = datum['label']
					print(k.decode(), x.shape, y)

def main():
	basic_operation()

	#--------------------
	# Usage:
	#	For using Caffe Datum:
	#		protoc --python_out=. caffe.proto

	use_caffe_datum = False
	#write_to_db_example(use_caffe_datum)
	#read_from_db_example(use_caffe_datum)
	#key_value_example(use_caffe_datum)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
