# REF [site] >> https://www.tensorflow.org/programmers_guide/graphs

import tensorflow as tf

print('[1] Session id =', id(tf.get_default_session()))
print('[1] Graph id =', id(tf.get_default_graph()))

config = tf.ConfigProto()

graph1 = tf.Graph()
graph2 = tf.Graph()

sess1 = tf.Session(graph=graph1, config=config)
sess2 = tf.Session(graph=graph2, config=config)

print('Session1 id =', id(sess1))
print('Graph1 id =', id(graph1))
print('Session2 id =', id(sess2))
print('Graph2 id =', id(graph2))

print('[2] Session id =', id(tf.get_default_session()))
print('[2] Graph id =', id(tf.get_default_graph()))

#with graph1.as_default() as graph:
with sess1:
	v1 = tf.get_variable('v1', shape=(3, 3))
	v2 = tf.get_variable('v2', shape=(2))

	c = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], name='c')

	x1 = tf.placeholder(tf.float32, shape=(2, 3), name='x1')
	x2 = tf.placeholder(tf.float32, shape=(2, 3), name='x2')
	A = tf.placeholder(tf.float32, shape=(5, 10), name='A')
	B = tf.placeholder(tf.float32, shape=(10, 3), name='B')

	y1 = tf.sqrt(x1, name='sqrt')
	y2 = tf.sin(x2, name='sin')
	z2 = tf.add(y1, y2, name='add')
	C = tf.matmul(A, B, name='matmul')

	print('Global variables =', tf.global_variables())
	print('Local variables =', tf.local_variables())
	print('Model variables =', tf.model_variables())
	#print('Operations =', graph.get_operations())

#with graph2.as_default() as graph:
with sess2:
	v1 = tf.get_variable('V1', shape=(3, 3))
	v2 = tf.get_variable('V2', shape=(2))

	c = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], name='c')

	x1 = tf.placeholder(tf.float32, shape=(2, 3), name='x1')
	x2 = tf.placeholder(tf.float32, shape=(2, 3), name='x2')
	A = tf.placeholder(tf.float32, shape=(5, 10), name='A')
	B = tf.placeholder(tf.float32, shape=(10, 3), name='B')

	y1 = tf.sqrt(x1, name='sqrt')
	y2 = tf.sin(x2, name='sin')
	z2 = tf.add(y1, y2, name='add')
	C = tf.matmul(A, B, name='matmul')

	print('Global variables =', tf.global_variables())
	print('Local variables =', tf.local_variables())
	print('Model variables =', tf.model_variables())
	#print('Operations =', graph.get_operations())

print('[3] Session id =', id(tf.get_default_session()))
print('[3] Graph id =', id(tf.get_default_graph()))
print('Global variables =', tf.global_variables())

with sess1:
	print('[4] Session id =', id(tf.get_default_session()))  # default session == sess1.
	print('[4] Graph id =', id(tf.get_default_graph()))  # default graph == graph1.
	print('Global variables =', tf.global_variables())
with sess2:
	print('[5] Session id =', id(tf.get_default_session()))  # default session == sess2.
	print('[5] Graph id =', id(tf.get_default_graph()))  # default graph == graph2.
	print('Global variables =', tf.global_variables())
	
with graph1.as_default():
	print('[6] Session id =', id(tf.get_default_session()))  # default session != sess1.
	print('[6] Graph id =', id(tf.get_default_graph()))  # default graph == graph1.
	print('Global variables =', tf.global_variables())
with graph2.as_default():
	print('[7] Session id =', id(tf.get_default_session()))  # default session != sess2.
	print('[7] Graph id =', id(tf.get_default_graph()))  # default graph == graph2.
	print('Global variables =', tf.global_variables())

print('[8] Session id =', id(tf.get_default_session()))
print('[8] Graph id =', id(tf.get_default_graph()))
