#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json
import numpy as np
#import caffe
import leveldb

def write_to_db_example(use_caffe_datum=False):
	N = 1000
	X = np.zeros((N, 3, 32, 32), dtype=np.uint8)
	y = np.zeros(N, dtype=np.int64)

	leveldb_dir_path = './myleveldb'
	db = leveldb.LevelDB(leveldb_dir_path, create_if_missing=True)
	if use_caffe_datum:
		#import caffe
		import caffe_pb2

		for i in range(N):
			# REF [site] >> https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto
			#datum = caffe.proto.caffe_pb2.Datum()
			datum = caffe_pb2.Datum()
			datum.channels = X.shape[1]
			datum.height = X.shape[2]
			datum.width = X.shape[3]
			datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9.
			datum.label = int(y[i])
			str_id = '{:08}'.format(i)

			# The encode is only essential in Python 3.
			db.Put(str_id.encode('ascii'), datum.SerializeToString())
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
			db.Put(str_id.encode('ascii'), json.dumps(datum).encode('ascii'))

	#db.Delete(b'00000000')

	#--------------------
	print(db.GetStats())

def read_from_db_example(use_caffe_datum=False):
	leveldb_dir_path = './myleveldb'
	db = leveldb.LevelDB(leveldb_dir_path, create_if_missing=True)
	key = b'00000000'
	try:
		raw_datum = db.Get(key)
	except KeyError as ex:
		print('Invalid key, {}.'.format(key))
		return

	if use_caffe_datum:
		#import caffe
		import caffe_pb2

		# REF [site] >> https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto
		#datum = caffe.proto.caffe_pb2.Datum()
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
	leveldb_dir_path = './myleveldb'
	db = leveldb.LevelDB(leveldb_dir_path, create_if_missing=True)
	if use_caffe_datum:
		#import caffe
		import caffe_pb2

		for k, v in db.RangeIter():
			# REF [site] >> https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto
			#datum = caffe.proto.caffe_pb2.Datum()
			datum = caffe_pb2.Datum()
			datum.ParseFromString(v)

			x = np.fromstring(datum.data, dtype=np.uint8)
			x = x.reshape(datum.channels, datum.height, datum.width)
			y = datum.label
			print(k.decode(), x.shape, y)
	else:
		for k, v in db.RangeIter():
			datum = json.loads(v.decode('ascii'))
			x = np.array(datum['data'], dtype=np.uint8)
			x = x.reshape(datum['channels'], datum['height'], datum['width'])
			y = datum['label']
			print(k.decode(), x.shape, y)

def main():
	# Usage:
	#	For using Caffe Datum:
	#		protoc --python_out=. caffe.proto
	use_caffe_datum = False
	write_to_db_example(use_caffe_datum)
	#read_from_db_example(use_caffe_datum)
	#key_value_example(use_caffe_datum)

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
