#!/usr/bin/env python

# NOTE [info] >>
#	https://github.com/pytransitions/transitions
#	pip install transitions

from transitions import Machine, State
from transitions.core import MachineError

class Matter(object):
	def enter_solid(self):
		print('Enter solid state.')
	def exit_solid(self):
		print('Exit solid state.')
	def enter_gas(self):
		print('Enter gas state.')
	def exit_gas(self):
		print('Exit gas state.')

def matter_fsm():
	# States.
	states=[
		State(name='solid', on_enter=['enter_solid'], on_exit=['exit_solid']),
		State(name='liquid', ignore_invalid_triggers=True),
		State(name='gas', on_enter=['enter_gas'], on_exit=['exit_gas']),
		State(name='plasma')
	]

	# Transitions.
	transitions = [
		{'trigger': 'melt', 'source': 'solid', 'dest': 'liquid'},
		{'trigger': 'evaporate', 'source': 'liquid', 'dest': 'gas'},
		{'trigger': 'sublimate', 'source': 'solid', 'dest': 'gas'},
		{'trigger': 'ionize', 'source': 'gas', 'dest': 'plasma'}
	]

	# Initialize.
	lump = Matter()
	fsm = Machine(model=lump, states=states, transitions=transitions, initial='liquid')
	#fsm = Machine(model=lump, states=states, transitions=transitions, initial='liquid', ignore_invalid_triggers=True)

	#-------------------------
	print('Current state = {}'.format(lump.state))

	lump.evaporate()
	print('Current state = {}'.format(lump.state))

	lump.trigger('ionize')
	print('Current state = {}'.format(lump.state))

	print('Triggers of solid state: {}'.format(fsm.get_triggers('solid')))

	#-------------------------
	fsm.set_state('solid')
	lump.sublimate()

	lump.to_gas()
	print('Current state = {}'.format(lump.state))

	#-------------------------
	lump.to_liquid()
	try:
		lump.melt()
	except MachineError as ex:
		print('Exception raised: {}'.format(ex))

	lump.to_gas()
	try:
		lump.melt()
	except MachineError as ex:
		print('Exception raised: {}'.format(ex))

class PingPong(object):
    pass

def pingpong_fsm():
	# States.
	states=['playerA', 'playerB']

	# Transitions.
	transitions = [
		{'trigger': 'ping', 'source': 'playerA', 'dest': 'playerB'},
		{'trigger': 'pong', 'source': 'playerB', 'dest': 'playerA'},
		{'trigger': 'pingpong', 'source': 'playerA', 'dest': 'playerB'},
		{'trigger': 'pingpong', 'source': 'playerB', 'dest': 'playerA'}
	]

	# Initialize.
	game = PingPong()
	fsm = Machine(game, states=states, transitions=transitions, initial='playerA')

	#-------------------------
	print('Current state = {}'.format(game.state))

	game.ping()
	print('Current state = {}'.format(game.state))
	game.pong()
	print('Current state = {}'.format(game.state))

	game.pingpong()
	print('Current state = {}'.format(game.state))
	game.pingpong()
	print('Current state = {}'.format(game.state))

def main():
	matter_fsm()
	#pingpong_fsm()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
