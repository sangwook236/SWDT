# REF [site] >> https://stackoverflow.com/questions/35919020/whats-the-difference-of-name-scope-and-a-variable-scope-in-tensorflow
# REF [site] >> https://www.tensorflow.org/programmers_guide/variables
# REF [site] >> https://www.tensorflow.org/api_guides/python/state_ops

import tensorflow as tf

# NOTE [infortant] >>
#	- Variables created by tf.get_variable() are managed in a different way that variables created by tf.Variable are managed. (?)
#	- The name scope is not applied to variables created by tf.get_variable().
#	- In tf.get_variable('scope/name'), the name of variable is 'scope/name', but not 'name' in a scope 'scope'. (?)

def create_variables(use_get_variable=True):
	if True == use_get_variable:
		tf.get_variable('Variable1', shape=(3, 3))  # name = 'Variable1'.
		# NOTE [error] >> Trying to share variable my_var_scope1/Variable1, but specified shape (5, 5) and found shape (3, 3).
		#tf.get_variable('Variable1', shape=(5, 5))
		tf.get_variable('Variable1')  # Shares an existing variable.
		tf.get_variable('Variable2', shape=(3, 3))  # name = 'Variable2'.

	tf.Variable(1)  # name = 'Variable'.
	tf.Variable(1)  # name = 'Variable_1'.
	tf.Variable(3, name='Variable1')  # name = 'Variable1_1'.
	tf.Variable(5, name='Variable1')  # name = 'Variable1_2'.
	tf.Variable(3, name='Variable2')  # name = 'Variable2_1'.
	tf.Variable(3, name='Variable3')  # name = 'Variable3'.

def create_operations():
	# REF [site] >> https://www.tensorflow.org/api_guides/python/math_ops

	x1 = tf.placeholder(tf.float32, shape=(2, 3))  # name = 'Placeholder'.
	x2 = tf.placeholder(tf.float32, shape=(3,))  # name = 'Placeholder_1'.
	A = tf.placeholder(tf.float32, shape=(5, 10))  # name = 'Placeholder_2'.

	y1 = tf.sqrt(x1)  # name = 'Sqrt'.
	y2 = tf.sin(x2)  # name = 'Sin'.
	z1 = tf.add(y1, y2)  # name = 'Add'.
	S1, U1, V1 = tf.svd(A, full_matrices=False, compute_uv=True)  # name = 'Svd'.

	x1 = tf.placeholder(tf.float32, shape=(2, 3), name='x1')  # name = 'x1'.
	x2 = tf.placeholder(tf.float32, shape=(3,), name='x2')  # name = 'x2'.
	A = tf.placeholder(tf.float32, shape=(5, 10), name='A')  # name = 'A'.

	y1 = tf.sqrt(x1, name='sqrt')  # name = 'sqrt'.
	y2 = tf.sin(x2, name='sin')  # name = 'sin'.
	z2 = tf.add(y1, y2, name='add')  # name = 'add'.
	S2, U2, V2 = tf.svd(A, full_matrices=False, compute_uv=True, name='svd')  # name = 'svd'.

	return z1, S1, U1, V1, z2, S2, U2, V2

#%%------------------------------------------------------------------
# Test re-using.

if False:
	with tf.variable_scope('my_scope1', reuse=tf.AUTO_REUSE):
		create_variables()
		create_operations()

	with tf.variable_scope('my_scope2', reuse=True):
		# NOTE [error] >>
		#	Variable my_scope2/Variable1 does not exist, or was not created with tf.get_variable().
		#	Did you mean to set reuse=tf.AUTO_REUSE in VarScope?
		#create_variables(True)
		create_variables(False)
		create_operations()

	with tf.variable_scope('my_scope3', reuse=None):
		# NOTE [error] >>
		#	Variable my_scope3/Variable1 already exists, disallowed.
		#	Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
		#create_variables(True)
		create_variables(False)
		create_operations()

#%%------------------------------------------------------------------

reuse = tf.AUTO_REUSE

with tf.variable_scope('my_var_scope1', reuse=reuse):
	create_variables()
	create_operations()

with tf.variable_scope('my_var_scope1', reuse=reuse):  # Its name is changed to 'my_var_scope1_1'.
	create_variables()
	create_operations()

with tf.variable_scope('my_var_scope2', reuse=reuse):
	create_variables()
	create_operations()

with tf.variable_scope('my_var_scope3', reuse=reuse):
	with tf.variable_scope('inner_var_scope', reuse=reuse):
		create_variables()
		create_operations()

with tf.variable_scope('my_var_scope4', reuse=reuse):
	with tf.name_scope('inner_name_scope'):  # NOTE [caution] >> The name scope is not applied to variables created by tf.get_variable().
		create_variables()
		create_operations()

#%%------------------------------------------------------------------

reuse = tf.AUTO_REUSE

with tf.name_scope('my_name_scope1'):  # NOTE [caution] >> The name scope is not applied to variables created by tf.get_variable().
	# NOTE [error] >>
	#	Variable Variable1 already exists, disallowed.
	#	Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
	#create_variables(True)
	create_variables(False)
	create_operations()

with tf.name_scope('my_name_scope1'):
	create_variables(False)
	create_operations()

with tf.name_scope('my_name_scope2'):
	create_variables(False)
	create_operations()

with tf.name_scope('my_name_scope3'):
	with tf.name_scope('inner_name_scope'):
		create_variables(False)
		create_operations()

with tf.name_scope('my_name_scope4'):
	with tf.variable_scope('inner_var_scope', reuse=reuse):
		create_variables(False)
		create_operations()

#%%------------------------------------------------------------------
# tf.Graph:
#	REF [site] >> https://www.tensorflow.org/api_docs/python/tf/Graph
#	A TensorFlow computation is represented as a dataflow graph.
#	A tf.Graph contains a set of tf.Operation objects, which represent units of computation, and tf.Tensor objects, which represent the units of data that flow between operations.
#	A default tf.Graph is always registered, and accessible by calling tf.get_default_graph().

graph = tf.get_default_graph()
#sess = tf.Session()
#graph = sess.graph
print(graph)

# The default graph is a property of the current thread.
# This function applies only to the current thread.
# Calling this function while a tf.Session or tf.InteractiveSession is active will result in undefined behavior.
# Using any previously created tf.Operation or tf.Tensor objects after calling this function will result in undefined behavior.
#tf.reset_default_graph()

#%%------------------------------------------------------------------
# tf.Operation:
#	REF [site] >> https://www.tensorflow.org/api_docs/python/tf/Operation
#	A tf.Operation represents a graph node that performs computation on tensors.
#	A tf.Operation is a node in a TensorFlow Graph that takes zero or more Tensor objects as input, and produces zero or more Tensor objects as output.

# <op_name> is like: scope1/scope2/.../scopen/name{_n}.

operations = graph.get_operations()
#print(operations)
for op in operations:
	#print(op)
	print(op.name)

operation = graph.get_operation_by_name('my_var_scope4/inner_name_scope/sqrt')  # Return an object of type 'tensorflow.python.framework.ops.Operation'.
print(operation)
print(operation.name)
print(operation.inputs)
print(operation.outputs)
print(operation.values())

#%%------------------------------------------------------------------
# tf.Tensor:
#	REF [site] >> https://www.tensorflow.org/api_docs/python/tf/Tensor
#	A tf.Tensor represents one of the outputs of a tf.Operation.
#	A tf.Tensor is a symbolic handle to one of the outputs of a tf.Operation.
#	It does not hold the values of that operation's output, but instead provides a means of computing those values in a TensorFlow tf.Session.

# <tensor_name> = <op_name>:<output_index>.

tensor = graph.get_tensor_by_name('my_var_scope4/inner_name_scope/sqrt:0')  # Return an object of type 'tensorflow.python.framework.ops.Tensor'.
#tensor = graph.get_tensor_by_name(operation.values()[0])  # tf.Operation.values() returns a tuple.
print(tensor)
print(tensor.name)
print(tensor.op)
print(tensor.get_shape())  # Returns a tf.TensorShape. Use tensor.get_shape().as_list().
#tensor.eval(feed_dict={...})

#%%------------------------------------------------------------------
# tf.Variable.
#	REF [site] >> https://www.tensorflow.org/api_docs/python/tf/Variable
#	A variable maintains state in the graph across calls to run().

global_variables = tf.global_variables()
#print(global_variables)
for var in global_variables:
	print(var)
local_variables = tf.local_variables()
#print(local_variables)
for var in local_variables:
	print(var)
model_variables = tf.model_variables()
#print(model_variables)
for var in model_variables:
	print(var)

#var = tf.get_variable('my_name_scope4/inner_var_scope/A')  # NOTE [error] >> Is not tf.Variable but tf.Tensor.
# NOTE [error] >>
#	 Variable my_var_scope1/Variable1 already exists, disallowed.
#	Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
#gvar = tf.get_variable('my_var_scope1/Variable1', shape=(5, 5, 1, 32))
#gvar = tf.get_variable('my_var_scope1/Variable1')

with tf.variable_scope('my_var_scope1', reuse=tf.AUTO_REUSE):
	# NOTE [error] >>
	#	Trying to share variable my_var_scope1/Variable1, but specified shape (3, 5) and found shape (3, 3).
	#gvar = tf.get_variable('Variable1', shape=(3, 5))

	gvar = tf.get_variable('Variable1', shape=(3, 3))  # Shares an existing variable.

	# A new variable ''my_var_scope1/Variable3_1' is created.
	gvar = tf.get_variable('Variable3', shape=(3, 5))
with tf.variable_scope('my_var_scope1', reuse=True):
	gvar = tf.get_variable('Variable1')  # Shares an existing variable.

	# NOTE [error] >> The variable 'my_var_scope1/Variable3' was not created with tf.get_variable(), but with tf.Variable().
	#	Variable my_var_scope1/Variable3 does not exist, or was not created with tf.get_variable().
	#	Did you mean to set reuse=tf.AUTO_REUSE in VarScope?
	#gvar = tf.get_variable('Variable3')
with tf.variable_scope('my_var_scope1', reuse=None):
	# NOTE [error] >>
	#	Variable my_var_scope1/Variable1 already exists, disallowed.
	#	Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope?
	#gvar = tf.get_variable('Variable1')

	pass

#%%------------------------------------------------------------------
# tf.VariableScope & tf.variable_scope:
#	REF [site] >> https://www.tensorflow.org/api_docs/python/tf/variable_scope
#	A context manager for defining ops that creates variables (layers).
#	This context manager validates that the (optional) values are from the same graph, ensures that graph is the default graph, and pushes a name scope and a variable scope.

curr_variable_scope = tf.get_variable_scope()  # Get the current variable scope.
print(curr_variable_scope)

#%%------------------------------------------------------------------
# tf.name_scope:
#	REF [site] >> https://www.tensorflow.org/api_docs/python/tf/name_scope
#	A context manager for use when defining a Python op.
#	This context manager validates that the given values are from the same graph, makes that graph the default graph, and pushes a name scope in that graph (see tf.Graph.name_scope for more details on that).

curr_name_scope = graph.get_name_scope()  # Get the current name scope.
print(curr_name_scope)
