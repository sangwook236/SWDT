# A python module that implements a Java interface to create a building object

import sys
sys.path.append('../jython')

from jython import IBuilding

class Building(IBuilding):
	def __init__(self, name, address, id):
		self.name = name
		self.address = address
		self.id = id

	def getBuildingName(self):
		return self.name

	def getBuildingAddress(self):
		return self.address

	def getBuildingId(self):
		return self.id

	def toString(self):
		return 'ID: {}, Name: {}, Address: {}'.format(self.id, self.name, self.address)
