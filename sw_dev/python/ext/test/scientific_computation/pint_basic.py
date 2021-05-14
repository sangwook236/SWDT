#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math, pint

def main():
	ureg = pint.UnitRegistry()

	def volume(radius, height):
		return math.pi * radius**2 * height

	vol = volume(3, 5) * ureg('cm^3')

	print('vol [cm^3] = {}.'.format(vol))
	print('vol [cubic inches] = {}.'.format(vol.to('cubic inches')))
	print('vol [gallons] = {}.'.format(vol.to('gallons').m))  # Magnitude.

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
