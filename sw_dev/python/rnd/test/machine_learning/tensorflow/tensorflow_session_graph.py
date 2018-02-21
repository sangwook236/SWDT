# REF [site] >> https://www.tensorflow.org/programmers_guide/graphs

import tensorflow as tf

print('[1] Session id =', id(tf.get_default_session()))
print('[1] Graph id =', id(tf.get_default_graph()))

config = tf.ConfigProto()

graph1 = tf.Graph()
graph2 = tf.Graph()
graph3 = tf.Graph()

with graph1.as_default() as graph:
	a1 = tf.get_variable('a1', shape=(3, 3))
	a2 = tf.get_variable('a2', shape=(2))

	c = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], name='c')

	x1 = tf.placeholder(tf.float32, shape=(2, 3), name='x1')
	x2 = tf.placeholder(tf.float32, shape=(2, 3), name='x2')
	y1 = tf.sqrt(x1, name='sqrt')
	y2 = tf.sin(x2, name='sin')
	z = tf.add(y1, y2, name='add')

	A = tf.placeholder(tf.float32, shape=(5, 10), name='A')
	B = tf.placeholder(tf.float32, shape=(10, 3), name='B')
	C = tf.matmul(A, B, name='matmul')

	print('Global variables =', tf.global_variables())
	print('Local variables =', tf.local_variables())
	print('Model variables =', tf.model_variables())
	#print('Operations =', graph.get_operations())

with graph3.as_default() as graph:
	c1 = tf.get_variable('c1', shape=(3, 3))

	c = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], name='c')

	x1 = tf.placeholder(tf.float32, shape=(2, 3), name='x1')
	x2 = tf.placeholder(tf.float32, shape=(2, 3), name='x2')
	y1 = tf.sqrt(x1, name='sqrt')
	y2 = tf.sin(x2, name='sin')
	z = tf.add(y1, y2, name='add')

	print('Global variables =', tf.global_variables())
	print('Local variables =', tf.local_variables())
	print('Model variables =', tf.model_variables())
	#print('Operations =', graph.get_operations())

sess1 = tf.Session(graph=graph1, config=config)
sess2 = tf.Session(graph=graph2, config=config)
sess3 = tf.Session(graph=graph3, config=config)

print('Session #1 id =', id(sess1))
print('Graph #1 id =', id(graph1))
print('Session #2 id =', id(sess2))
print('Graph #2 id =', id(graph2))
print('Session #3 id =', id(sess3))
print('Graph #3 id =', id(graph3))

print('[2] Session id =', id(tf.get_default_session()))
print('[2] Graph id =', id(tf.get_default_graph()))

#with graph2.as_default() as graph:
with sess2:
	b1 = tf.get_variable('b1', shape=(3, 3))
	b2 = tf.get_variable('b2', shape=(2))

	c = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], name='c')

	x1 = tf.placeholder(tf.float32, shape=(2, 3), name='x1')
	x2 = tf.placeholder(tf.float32, shape=(2, 3), name='x2')
	y1 = tf.sqrt(x1, name='sqrt')
	y2 = tf.sin(x2, name='sin')
	z = tf.add(y1, y2, name='add')

	A = tf.placeholder(tf.float32, shape=(5, 10), name='A')
	B = tf.placeholder(tf.float32, shape=(10, 3), name='B')
	C = tf.matmul(A, B, name='matmul')

	print('Global variables =', tf.global_variables())
	print('Local variables =', tf.local_variables())
	print('Model variables =', tf.model_variables())
	#print('Operations =', graph.get_operations())

#with graph3.as_default() as graph:
with sess3:
	c2 = tf.get_variable('c2', shape=(2))

	A = tf.placeholder(tf.float32, shape=(5, 10), name='A')
	B = tf.placeholder(tf.float32, shape=(10, 3), name='B')
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
with sess3:
	print('[6] Session id =', id(tf.get_default_session()))  # default session == sess32.
	print('[6] Graph id =', id(tf.get_default_graph()))  # default graph == graph3.
	print('Global variables =', tf.global_variables())
	
with graph1.as_default():
	print('[7] Session id =', id(tf.get_default_session()))  # default session != sess1.
	print('[7] Graph id =', id(tf.get_default_graph()))  # default graph == graph1.
	print('Global variables =', tf.global_variables())
with graph2.as_default():
	print('[8] Session id =', id(tf.get_default_session()))  # default session != sess2.
	print('[8] Graph id =', id(tf.get_default_graph()))  # default graph == graph2.
	print('Global variables =', tf.global_variables())
with graph3.as_default():
	print('[9] Session id =', id(tf.get_default_session()))  # default session != sess32.
	print('[9] Graph id =', id(tf.get_default_graph()))  # default graph == graph32.
	print('Global variables =', tf.global_variables())

print('[10] Session id =', id(tf.get_default_session()))
print('[10] Graph id =', id(tf.get_default_graph()))
