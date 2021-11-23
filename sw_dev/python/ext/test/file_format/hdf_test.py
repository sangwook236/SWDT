#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import h5py

# REF [site] >> https://docs.h5py.org/en/stable/quick.html
def quick_start_guide():
	# An HDF5 file is a container for two kinds of objects:
	#	datasets, which are array-like collections of data, and groups, which are folder-like containers that hold datasets and other groups.
	# The most fundamental thing to remember when using h5py is:
	#	Groups work like dictionaries, and datasets work like NumPy arrays.

	hdf5_filepath = "./mytestfile.hdf5"

	# Create a file.
	with h5py.File(hdf5_filepath, "w") as f:
		dset = f.create_dataset("mydataset", shape=(100,), dtype="i")

		print("f.name = {}.".format(f.name))
		print("dset.name = {}.".format(dset.name))

		# Attribute.
		#	The official way to store metadata in HDF5.
		dset.attrs["temperature"] = 99.5
		print('dset.attrs["temperature"] = {}.'.format(dset.attrs["temperature"]))
		print('"temperature" in dset.attrs = {}.'.format("temperature" in dset.attrs))

	with h5py.File(hdf5_filepath, "a") as f:
		grp = f.create_group("subgroup")

		dset2 = grp.create_dataset("another_dataset", shape=(50,), dtype="f")
		print("dset2.name = {}.".format(dset2.name))

		dset3 = f.create_dataset("subgroup2/dataset_three", shape=(10,), dtype="i")
		print("dset3.name = {}.".format(dset3.name))

		for name in f:
			print(name)

		print('"mydataset" in f = {}.'.format("mydataset" in f))
		print('"somethingelse" in f = {}.'.format("somethingelse" in f))
		print('"subgroup/another_dataset" in f = {}.'.format("subgroup/another_dataset" in f))

		print("f.keys() = {}.".format(f.keys()))
		print("f.values() = {}.".format(f.values()))
		print("f.items() = {}.".format(f.items()))
		#print("f.iter() = {}.".format(f.iter()))  # AttributeError: 'File' object has no attribute 'iter'.
		print('f.get("subgroup/another_dataset") = {}.'.format(f.get("subgroup/another_dataset")))
		print('f.get("another_dataset") = {}.'.format(f.get("another_dataset")))
		print('f["subgroup/another_dataset"] = {}.'.format(f["subgroup/another_dataset"]))
		#print('f["another_dataset"] = {}.'.format(f["another_dataset"]))  # KeyError: "Unable to open object (object 'another_dataset' doesn't exist)".
		dataset_three = f["subgroup2/dataset_three"]

		print("grp.keys() = {}.".format(grp.keys()))
		print("grp.values() = {}.".format(grp.values()))
		print("grp.items() = {}.".format(grp.items()))
		#print("grp.iter() = {}.".format(grp.iter()))  # AttributeError: 'Group' object has no attribute 'iter'.
		print('grp.get("another_dataset") = {}.'.format(grp.get("another_dataset")))
		print('grp.get("subgroup/another_dataset") = {}.'.format(grp.get("subgroup/another_dataset")))
		print('grp["another_dataset"] = {}.'.format(grp["another_dataset"]))
		#print('grp["subgroup/another_dataset"] = {}.'.format(grp["subgroup/another_dataset"]))  # KeyError: 'Unable to open object (component not found)'.

		del grp["another_dataset"]

		def print_name(name):
			print(name)

		f.visit(print_name)

		def print_item(name, obj):
			print(name, obj)

		f.visititems(print_item)

	with h5py.File(hdf5_filepath, "r+") as f:
		dset = f["mydataset"]
		print("dset.shape = {}, dset.dtype= {}.".format(dset.shape, dset.dtype))

		dset[...] = np.arange(100)
		print("dset[0] = {}.".format(dset[0]))
		print("dset[10] = {}.".format(dset[10]))
		print("dset[0:100:10] = {}.".format(dset[0:100:10]))

def main():
	quick_start_guide()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
  