# REF [site] >> https://reinforce.io/blog/introduction-to-tensorforce/
# REF [site] >> https://reinforce.io/blog/end-to-end-computation-graphs-for-reinforcement-learning/

# Path to libcudnn.so.
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#--------------------
import os, sys
if 'posix' == os.name:
	swl_python_home_dir_path = '/home/sangwook/work/SWL_github/python'
	lib_home_dir_path = '/home/sangwook/lib_repo/python'
else:
	swl_python_home_dir_path = 'D:/work/SWL_github/python'
	#lib_home_dir_path = 'D:/lib_repo/python'
	lib_home_dir_path = 'D:/lib_repo/python/rnd'
sys.path.append(swl_python_home_dir_path + '/src')
sys.path.append(lib_home_dir_path + '/tensorforce_github')
#sys.path.append(lib_home_dir_path + '/gym_github')

#---------------------------------------------------------------------

from tensorforce.agents import DQNAgent

# Network is an ordered list of layers.
network_spec = [
	dict(type='dense', size=32, activation='tanh'),
	dict(type='dense', size=32, activation='tanh')
]

# Define a state.
states = dict(shape=(10,), type='float')
#states = dict(
#	image=dict(shape=(64, 64, 3), type='float'),
#	caption=dict(shape=(20,), type='int')
#)

# Define an action.
actions = dict(type='int', num_actions=5)

# The agent is configured with a single configuration object.
config = dict(
	memory=dict(type='replay', capacity=1000),
	batch_size=8,
	first_update=100,
	target_sync_frequency=10
)

agent = DQNAgent(
	states_spec=states,
	actions_spec=actions,
	network_spec=network_spec,
	**config
)
