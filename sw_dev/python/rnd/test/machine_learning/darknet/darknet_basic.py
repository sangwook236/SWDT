#!/usr/bin/env python

# export LD_LIBRARY_PATH+=${DARKNET_HOME}
#	or copy libdarknet.so to the working directory.
# export PYTHONPATH+=${DARKNET_HOME}/python

import os
import darknet

def main():
	if 'posix' == os.name:
		darknet_home_dir_path = '/home/sangwook/lib_repo/cpp/darknet_github'
	else:
		darknet_home_dir_path = 'D:/lib_repo/cpp/rnd/darknet_github'
		#darknet_home_dir_path = 'D:/lib_repo/cpp/darknet_github'

	net = darknet.load_net(bytes(darknet_home_dir_path + '/cfg/yolov3.cfg', encoding='utf-8'), bytes(darknet_home_dir_path + '/yolov3.weights', encoding='utf-8'), 0)
	meta = darknet.load_meta(bytes(darknet_home_dir_path + '/cfg/coco.data', encoding='utf-8'))
	detections = darknet.detect(net, meta, bytes(darknet_home_dir_path + '/data/dog.jpg', encoding='utf-8'))
	print('Detections =', detections)

#%%-------------------------------------------------------------------

if '__main__' == __name__:
	main()
