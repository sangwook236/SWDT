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
	a3 = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], name='a3')
	a4 = tf.placeholder(tf.float32, shape=(2, 3), name='a4')

	x1_1 = tf.constant(1, dtype=tf.float32, shape=[2, 3], name='x1_1')
	x1_2 = tf.constant(2, dtype=tf.float32, shape=[2, 3], name='x1_2')
	y1_1 = tf.sqrt(x1_1, name='y1_1')
	y1_2 = tf.sin(x1_2, name='y1_2')
	z1 = tf.add(y1_1, y1_2, name='z1')

	A1 = tf.constant(1, dtype=tf.float32, shape=(5, 10), name='A1')
	B1 = tf.constant(2, dtype=tf.float32, shape=(10, 3), name='B1')
	C1 = tf.matmul(A1, B1, name='C1')

	print('Global variables =', tf.global_variables())
	print('Local variables =', tf.local_variables())
	print('Model variables =', tf.model_variables())
	#print('Operations =', graph.get_operations())

with graph3.as_default() as graph:
	c1 = tf.get_variable('c1', shape=(3, 3))
	c3 = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], name='c3')

	x3_1 = tf.constant(5, dtype=tf.float32, shape=[2, 3], name='x3_1')
	x3_2 = tf.constant(6, dtype=tf.float32, shape=[2, 3], name='x3_2')
	y3_1 = tf.sqrt(x3_1, name='y3_1')
	y3_2 = tf.sin(x3_2, name='y3_2')
	z3 = tf.add(y3_1, y3_2, name='z3')

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

with graph2.as_default() as graph:
#with sess2:
	b1 = tf.get_variable('b1', shape=(3, 3))
	b2 = tf.get_variable('b2', shape=(2))
	b3 = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], name='b3')
	b4 = tf.placeholder(tf.float32, shape=(2, 3), name='b4')

	x2_1 = tf.constant(3, dtype=tf.float32, shape=[2, 3], name='x2_1')
	x2_2 = tf.constant(4, dtype=tf.float32, shape=[2, 3], name='x2_2')
	y2_1 = tf.sqrt(x2_1, name='y2_1')
	y2_2 = tf.sin(x2_2, name='y2_2')
	z2 = tf.add(y2_1, y2_2, name='z2')

	A2 = tf.constant(3, dtype=tf.float32, shape=(5, 10), name='A2')
	B2 = tf.constant(4, dtype=tf.float32, shape=(10, 3), name='B2')
	C2 = tf.matmul(A2, B2, name='C2')

	print('Global variables =', tf.global_variables())
	print('Local variables =', tf.local_variables())
	print('Model variables =', tf.model_variables())
	#print('Operations =', graph.get_operations())

with graph3.as_default() as graph:
#with sess3:
	c2 = tf.get_variable('c2', shape=(2))
	c3 = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], name='c3')

	A3 = tf.constant(5, dtype=tf.float32, shape=(5, 10), name='A3')
	B3 = tf.constant(6, dtype=tf.float32, shape=(10, 3), name='B3')
	C3 = tf.matmul(A3, B3, name='C3')

	print('Global variables =', tf.global_variables())
	print('Local variables =', tf.local_variables())
	print('Model variables =', tf.model_variables())
	#print('Operations =', graph.get_operations())

print('[3] Session id =', id(tf.get_default_session()))
print('[3] Graph id =', id(tf.get_default_graph()))
print('Global variables =', tf.global_variables())

with sess1 as sess:
	print('[4] Session id =', id(tf.get_default_session()))  # default session == sess1.
	print('[4] Graph id =', id(tf.get_default_graph()))  # default graph == graph1.
	print('Global variables =', tf.global_variables())
with sess2 as sess:
	print('[5] Session id =', id(tf.get_default_session()))  # default session == sess2.
	print('[5] Graph id =', id(tf.get_default_graph()))  # default graph == graph2.
	print('Global variables =', tf.global_variables())
with sess3 as sess:
	print('[6] Session id =', id(tf.get_default_session()))  # default session == sess3.
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
	print('[9] Session id =', id(tf.get_default_session()))  # default session != sess3.
	print('[9] Graph id =', id(tf.get_default_graph()))  # default graph == graph3.
	print('Global variables =', tf.global_variables())

print('[10] Session id =', id(tf.get_default_session()))
print('[10] Graph id =', id(tf.get_default_graph()))

#%%------------------------------------------------------------------

sess1.close()  # Error: Don't close the session.
print('C1 =', sess1.run(C1))

print('C2 =', C2.eval(session=sess2))
print('C3 =', sess3.run(C3))

with sess3.as_default() as sess:
	#print(C3.eval(session=sess))
	print('C3 =', sess.run(C3))

"""
# NOTICE [caution] >> The session will automatically be closed when the with block is exited.
#	If tf.Tensor.eval() or tf.Session.run() is called in the with block, the session (sess3) will be closed in the end of the with block.
#	But sometimes the session is not closed. => I do not know why.
with sess3 as sess:
	#print(C3.eval(session=sess))
	print('C3 =', sess.run(C3))
"""

print('C2 =', sess2.run(C2))
print('C3 =', C3.eval(session=sess3))  # Run-time error: Attempted to use a closed Session.

with tf.Session(graph=graph3) as sess:
	print('Global variables =', tf.global_variables())
	print('C3 =', sess.run(C3))
