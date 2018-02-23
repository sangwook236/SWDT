# REF [site] >> https://www.tensorflow.org/programmers_guide/graphs

import tensorflow as tf

default_sess = tf.get_default_session()
default_graph = tf.get_default_graph()

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

config = tf.ConfigProto()
sess1 = tf.Session(graph=graph1, config=config)
sess2 = tf.Session(graph=graph2, config=config)
sess3 = tf.Session(graph=graph3, config=config)

with graph2.as_default() as graph:
#with sess2 as sess:
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
#with sess3 as sess:
	c2 = tf.get_variable('c2', shape=(2))
	c3 = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], name='c3')

	A3 = tf.constant(5, dtype=tf.float32, shape=(5, 10), name='A3')
	B3 = tf.constant(6, dtype=tf.float32, shape=(10, 3), name='B3')
	C3 = tf.matmul(A3, B3, name='C3')

	print('Global variables =', tf.global_variables())
	print('Local variables =', tf.local_variables())
	print('Model variables =', tf.model_variables())
	#print('Operations =', graph.get_operations())

#%%------------------------------------------------------------------

assert tf.get_default_session() is default_sess
assert tf.get_default_graph() is default_graph
print('[0] Global variables =', tf.global_variables())

with sess1 as sess:
	assert sess is sess1
	assert sess.graph is graph1
	assert tf.get_default_session() is sess  # default session == sess1.
	assert tf.get_default_graph() is sess.graph  # default graph == graph1.
	print('[1-1] Global variables =', tf.global_variables())
with sess2 as sess:
	assert sess is sess2
	assert sess.graph is graph2
	assert tf.get_default_session() is sess  # default session == sess2.
	assert tf.get_default_graph() is sess.graph  # default graph == graph2.
	print('[1-2] Global variables =', tf.global_variables())
with sess3 as sess:
	assert sess is sess3
	assert sess.graph is graph3
	assert tf.get_default_session() is sess  # default session == sess3.
	assert tf.get_default_graph() is sess.graph  # default graph == graph3.
	print('[1-3] Global variables =', tf.global_variables())

# NOTICE [info] >>
#	Entering a with sess.as_default(): block does not affect the current default graph.
#	If you are using multiple graphs, and sess.graph is different from the value of tf.get_default_graph(), you must explicitly enter a with sess.graph.as_default(): block to make sess.graph the default graph.
with sess1.as_default() as sess:
	assert sess is sess1
	assert sess.graph is graph1
	assert tf.get_default_session() is sess  # default session == sess1.
	assert tf.get_default_graph() is not sess.graph  # default graph != graph1.
	print('[2-1] Global variables =', tf.global_variables())
with sess2.as_default() as sess:
	assert sess is sess2
	assert sess.graph is graph2
	assert tf.get_default_session() is sess  # default session == sess2.
	assert tf.get_default_graph() is not sess.graph  # default graph != graph2.
	print('[2-2] Global variables =', tf.global_variables())
with sess3.as_default() as sess:
	assert sess is sess3
	assert sess.graph is graph3
	assert tf.get_default_session() is sess  # default session == sess3.
	assert tf.get_default_graph() is not sess.graph  # default graph != graph3.
	print('[2-3] Global variables =', tf.global_variables())

with sess1.as_default() as sess:
	with sess.graph.as_default() as graph:
		assert sess is sess1
		assert graph is graph1
		assert sess.graph is graph1
		assert tf.get_default_session() is sess  # default session == sess1.
		assert tf.get_default_graph() is sess.graph  # default graph == graph1.
		print('[3-1] Global variables =', tf.global_variables())
with sess2.as_default() as sess:
	with sess.graph.as_default() as graph:
		assert sess is sess2
		assert graph is graph2
		assert sess.graph is graph2
		assert tf.get_default_session() is sess  # default session == sess2.
		assert tf.get_default_graph() is sess.graph  # default graph == graph2.
		print('[3-2] Global variables =', tf.global_variables())
with sess3.as_default() as sess:
	with sess.graph.as_default() as graph:
		assert sess is sess3
		assert graph is graph3
		assert sess.graph is graph3
		assert tf.get_default_session() is sess  # default session == sess3.
		assert tf.get_default_graph() is sess.graph  # default graph == graph3.
		print('[3-3] Global variables =', tf.global_variables())

# NOTICE [info] >>
#	Entering a with graph.as_default(): block does not affect the current default session.
with graph1.as_default() as graph:
	assert graph is graph1
	assert tf.get_default_session() is not sess1  # default session != sess1.
	assert tf.get_default_graph() is graph1  # default graph == graph1.
	print('[4-1] Global variables =', tf.global_variables())
with graph2.as_default() as graph:
	assert graph is graph2
	assert tf.get_default_session() is not sess2  # default session != sess2.
	assert tf.get_default_graph() is graph2  # default graph == graph2.
	print('[4-2] Global variables =', tf.global_variables())
with graph3.as_default() as graph:
	assert graph is graph3
	assert tf.get_default_session() is not sess3  # default session != sess3.
	assert tf.get_default_graph() is graph3  # default graph == graph3.
	print('[4-3] Global variables =', tf.global_variables())

assert tf.get_default_session() is default_sess
assert tf.get_default_graph() is default_graph

#%%------------------------------------------------------------------

# NOTICE [info] >>
#	The as_default context manager does not close the session (sess1) when you exit the context, and it must be closed explicitly.
with sess1.as_default() as sess:
	#print('C1 =', C1.eval(session=sess))
	print('C1 =', sess.run(C1))

# NOTICE [info] >> Uses a (temporary) session.
with tf.Session(graph=graph1) as sess:
	print('Global variables =', tf.global_variables())
	#print('C1 =', C1.eval(session=sess))
	print('C1 =', sess.run(C1))

print('C1 =', sess1.run(C1))
sess1.close()
#sess1 = None  # A good practice to prevent misuse of a closed session.
#print('C1 =', sess1.run(C1))  # Run-time error: Attempted to use a closed Session.

# NOTICE [caution] >> When the with block is exited, the session (sess1) will automatically be closed.
#	But sometimes the session is not closed. => I do not know why.
with sess1 as sess:
	# NOTICE [info] >> The sess1 instance still exists even if it is already closed.
	assert tf.get_default_session() is sess

	# Run-time error: Attempted to use a closed Session.
	#print('C1 =', C1.eval(session=sess))
	#print('C1 =', sess.run(C1))
#sess1 = None  # A good practice to prevent misuse of a closed session.

with sess1.as_default() as sess:
	# NOTICE [info] >> The sess1 instance still exists even if it is already closed.
	assert tf.get_default_session() is sess

assert tf.get_default_session() is not sess1

#%%------------------------------------------------------------------

sess2.close()
#sess2 = None
sess3.close()
#sess3 = None
