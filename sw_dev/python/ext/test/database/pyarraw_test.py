#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pyarrow as pa

# REF [site] >> https://arrow.apache.org/docs/python/memory.html
def memory_and_io_interfaces_example():
	# pyarrow.Buffer.

	data = b"abcdefghijklmnopqrstuvwxyz"

	# Creating a Buffer in this way does not allocate any memory; it is a zero-copy view on the memory exported from the data bytes object.
	buf = pa.py_buffer(data)
	# External memory, under the form of a raw pointer and size, can also be referenced using the foreign_buffer() function.
	#buf = pa.foreign_buffer(data)

	print("buf = {}.".format(buf))
	print("buf.size = {}.".format(buf.size))

	print("memoryview(buf) = {}.".format(memoryview(buf)))
	print("buf.to_pybytes() = {}.".format(buf.to_pybytes()))

	#--------------------
	# Memory pools.

	print("pa.total_allocated_bytes() = {}.".format(pa.total_allocated_bytes()))

	buf = pa.allocate_buffer(1024, resizable=True)
	print("pa.total_allocated_bytes() = {}.".format(pa.total_allocated_bytes()))

	buf.resize(2048)
	print("pa.total_allocated_bytes() = {}.".format(pa.total_allocated_bytes()))

	buf = None
	print("pa.total_allocated_bytes() = {}.".format(pa.total_allocated_bytes()))

	print("pa.default_memory_pool().backend_name = {}.".format(pa.default_memory_pool().backend_name))

	#--------------------
	# Input and output streams.

	buf = memoryview(b"some data")
	stream = pa.input_stream(buf)

	print("stream.read(4) = {}.".format(stream.read(4)))

	import gzip
	with gzip.open("./example.gz", "wb") as f:
		f.write(b"some data\n" * 3)

	stream = pa.input_stream("./example.gz")
	print("stream.read() = {}.".format(stream.read()))

	with pa.output_stream("./example1.dat") as stream:
		stream.write(b"some data")

	f = open("./example1.dat", "rb")
	print("f.read() = {}.".format(f.read()))

	#--------------------
	# On-disk and memory mapped files.

	# Using regular Python.
	with open("./example2.dat", "wb") as f:
		f.write(b"some example data")

	file_obj = pa.OSFile("./example2.dat")
	print("file_obj.read(4) = {}.".format(file_obj.read(4)))

	# Using pyarrow's OSFile class.
	with pa.OSFile("./example3.dat", "wb") as f:
		f.write(b"some example data")

	mmap = pa.memory_map("./example3.dat")
	print("mmap.read(4) = {}.".format(mmap.read(4)))

	mmap.seek(0)
	buf = mmap.read_buffer(4)
	print("buf = {}.".format(buf))
	print("buf.to_pybytes() = {}.".format(buf.to_pybytes()))

	#--------------------
	# In-memory reading and writing.

	writer = pa.BufferOutputStream()
	writer.write(b"hello, friends")
	buf = writer.getvalue()
	print("buf = {}.".format(buf))
	print("buf.size = {}.".format(buf.size))

	reader = pa.BufferReader(buf)
	reader.seek(7)
	print("reader.read(7) = {}.".format(reader.read(7)))

# REF [site] >> https://arrow.apache.org/docs/python/data.html
def data_types_and_in_memory_data_model_example():
	# Type metadata.
	t1 = pa.int32()
	t2 = pa.string()
	t3 = pa.binary()
	t4 = pa.binary(10)
	t5 = pa.timestamp("ms")

	print("t1 = {}.".format(t1))
	print("t4 = {}.".format(t4))
	print("t5 = {}.".format(t5))

	f0 = pa.field("int32_field", t1)
	print("f0 = {}.".format(f0))
	print("f0.name = {}.".format(f0.name))
	print("f0.type = {}.".format(f0.type))

	t6 = pa.list_(t1)
	print("t6 = {}.".format(t6))

	fields = [
		pa.field("s0", t1),
		pa.field("s1", t2),
		pa.field("s2", t4),
		pa.field("s3", t6),
	]
	t7 = pa.struct(fields)
	print("t7 = {}.".format(t7))

	t8 = pa.struct([("s0", t1), ("s1", t2), ("s2", t4), ("s3", t6)])
	print("t8 = {}.".format(t8))

	print("'t8 == t7' = {}.".format(t8 == t7))

	# Schemas.
	my_schema = pa.schema([
		("field0", t1),
		("field1", t2),
		("field2", t4),
		("field3", t6)
	])
	print("my_schema = {}.".format(my_schema))

	# Arrays.
	arr = pa.array([1, 2, None, 3])
	#arr = pa.array([1, 2, None, 3], type=pa.uint16())

	print("arr = {}.".format(arr))
	print("arr.type = {}.".format(arr.type))
	print("len(arr) = {}.".format(len(arr)))
	print("arr.null_count = {}.".format(arr.null_count))
	print("arr[0] = {}.".format(arr[0]))
	print("arr[2] = {}.".format(arr[2]))
	print("arr[1:3] = {}.".format(arr[1:3]))

	# List arrays.
	nested_arr = pa.array([[], None, [1, 2], [None, 1]])
	print("nested_arr.type = {}.".format(nested_arr.type))

	# Struct arrays.
	ty = pa.struct([("x", pa.int8()), ("y", pa.bool_())])
	print(pa.array([{"x": 1, "y": True}, {"x": 2, "y": False}], type=ty))
	#print('pa.array([{"x": 1, "y": True}, {"x": 2, "y": False}], type=ty) = {}.'.format(pa.array([{"x": 1, "y": True}, {"x": 2, "y": False}], type=ty)))  # KeyError: '"x"'.
	print(pa.array([(3, True), (4, False)], type=ty))
	#print("pa.array([(3, True), (4, False)], type=ty) = {}.".format(pa.array([(3, True), (4, False)], type=ty)))
	print(pa.array([{"x": 1}, None, {"y": None}], type=ty))

	xs = pa.array([5, 6, 7], type=pa.int16())
	ys = pa.array([False, True, True])
	arr = pa.StructArray.from_arrays((xs, ys), names=("x", "y"))
	print("arr.type = {}.".format(arr.type))
	print("arr = {}.".format(arr))

	# Union arrays.
	xs = pa.array([5, 6, 7])
	ys = pa.array([False, False, True])
	types = pa.array([0, 1, 1], type=pa.int8())
	union_arr = pa.UnionArray.from_sparse(types, [xs, ys])
	print("union_arr.type = {}.".format(union_arr.type))
	print("union_arr = {}.".format(union_arr))

	xs = pa.array([5, 6, 7])
	ys = pa.array([False, True])
	types = pa.array([0, 1, 1, 0, 0], type=pa.int8())
	offsets = pa.array([0, 0, 1, 1, 2], type=pa.int32())
	union_arr = pa.UnionArray.from_dense(types, offsets, [xs, ys])
	print("union_arr.type = {}.".format(union_arr.type))
	print("union_arr = {}.".format(union_arr))

	# Dictionary arrays.
	indices = pa.array([0, 1, 0, 1, 2, 0, None, 2])
	dictionary = pa.array(["foo", "bar", "baz"])
	dict_array = pa.DictionaryArray.from_arrays(indices, dictionary)
	print("dict_array = {}.".format(dict_array))
	print("dict_array.type = {}.".format(dict_array.type))
	print("dict_array.indices = {}.".format(dict_array.indices))
	print("dict_array.dictionary = {}.".format(dict_array.dictionary))
	print("dict_array.to_pandas() = {}.".format(dict_array.to_pandas()))

	# Record batches.
	#	A Record Batch in Apache Arrow is a collection of equal-length array instances.
	data = [
		pa.array([1, 2, 3, 4]),
		pa.array(["foo", "bar", "baz", None]),
		pa.array([True, None, False, True])
	]
	batch = pa.RecordBatch.from_arrays(data, ["f0", "f1", "f2"])
	print("batch.num_columns = {}.".format(batch.num_columns))
	print("batch.num_rows = {}.".format(batch.num_rows))
	print("batch.schema = {}.".format(batch.schema))
	print("batch[1] = {}.".format(batch[1]))
	batch2 = batch.slice(1, 3)
	print("batch2[1] = {}.".format(batch2[1]))

	# Tables.
	batches = [batch] * 5
	table = pa.Table.from_batches(batches)
	print("table = {}.".format(table))
	print("table.num_rows = {}.".format(table.num_rows))

	# The table's columns are instances of ChunkedArray, which is a container for one or more arrays of the same type.
	c = table[0]
	print("c = {}.".format(c))
	print("c.num_chunks = {}.".format(c.num_chunks))
	print("c.chunk(0) = {}.".format(c.chunk(0)))
	print("c.to_pandas() = {}.".format(c.to_pandas()))

	# Multiple tables can also be concatenated together to form a single table using pyarrow.concat_tables, if the schemas are equal.
	tables = [table] * 2
	table_all = pa.concat_tables(tables)
	print("table_all.num_rows = {}.".format(table_all.num_rows))
	c = table_all[0]
	print("c.num_chunks = {}.".format(c.num_chunks))

# REF [site] >> https://arrow.apache.org/docs/python/compute.html
def compute_functions_example():
	import pyarrow.compute

	a = pa.array([1, 1, 2, 3])
	pa.compute.sum(a)
	print("pa.compute.sum(a) = {}.".format(pa.compute.sum(a)))

	b = pa.array([4, 1, 2, 8])
	print("pa.compute.equal(a, b) = {}.".format(pa.compute.equal(a, b)))

	x, y = pa.scalar(7.8), pa.scalar(9.3)
	print("pa.compute.multiply(x, y) = {}.".format(pa.compute.multiply(x, y)))

def main():
	#memory_and_io_interfaces_example()
	data_types_and_in_memory_data_model_example()
	#compute_functions_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
