#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	http://opennmt.net/
#	https://github.com/OpenNMT/OpenNMT-py

import torch
import onmt

# REF [file] >> ${OpenNMT-py_HOME}/onmt/bin/preprocess.py
def preprocess_test():
	raise NotImplementedError

# REF [file] >> ${OpenNMT-py_HOME}/onmt/bin/train.py
def train_test():
	raise NotImplementedError

# REF [file] >> ${OpenNMT-py_HOME}/onmt/bin/translate.py
def translate_test():
	raise NotImplementedError

# REF [file] >> ${OpenNMT-py_HOME}/onmt/bin/server.py
def server_test():
	raise NotImplementedError

def main():
	preprocess_test()  # Not yet implemented.
	train_test()  # Not yet implemented.
	translate_test()  # Not yet implemented.
	server_test()  # Not yet implemented.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
