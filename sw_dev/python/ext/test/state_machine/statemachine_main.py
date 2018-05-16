#!/usr/bin/env python

from statemachine import StateMachine, State
from statemachine.exceptions import TransitionNotAllowed

class TrafficLightMachine(StateMachine):
	# States.
	green = State('Green', initial=True)
	yellow = State('Yellow')
	red = State('Red')

	# Transitions.
	slowdown = green.to(yellow)
	stop = yellow.to(red)
	go = red.to(green)

	cycle = green.to(yellow) | yellow.to(red) | red.to(green)

	def on_enter_green(self):
	    print('Enter green.')

	def on_exit_green(self):
	    print('Exit green.')

	def on_enter_yellow(self):
	    print('Enter yellow.')

	def on_exit_yellow(self):
	    print('Exit yellow.')

	def on_enter_red(self):
	    print('Enter red.')

	def on_exit_red(self):
	    print('Exit red.')

	def on_cycle(self):
	    print('Cycle.')

	def on_slowdown(self):
	    print('Slowdown.')

	def on_stop(self):
	    print('Stop.')

	def on_go(self):
	    print('Go.')

class MyModel(object):
	def __init__(self, state):
		self.state = state

def main():
	statemachine = TrafficLightMachine()
	#statemachine = TrafficLightMachine(start_value='red')
	#obj = MyModel(state='green')
	#statemachine = TrafficLightMachine(obj)

	try:
		statemachine.cycle()
		statemachine.cycle()
		statemachine.cycle()
	except TransitionNotAllowed as ex:
		print('TransitionNotAllowed:', ex)

	try:
		statemachine.slowdown()
		statemachine.stop()
		statemachine.go()
	except TransitionNotAllowed as ex:
		print('TransitionNotAllowed:', ex)

	try:
		statemachine.run('slowdown')
		statemachine.run('stop')
		statemachine.run('go')
	except TransitionNotAllowed as ex:
		print('TransitionNotAllowed:', ex)

	print(statemachine.current_state)
	print(statemachine.is_green)
	print(statemachine.is_yellow)
	print(statemachine.is_red)
	#print(obj.state)

	print([s.identifier for s in statemachine.states])
	print([t.identifier for t in statemachine.transitions])

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
