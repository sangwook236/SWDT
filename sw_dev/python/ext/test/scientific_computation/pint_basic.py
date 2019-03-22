#!/usr/bin/env python

import math, pint

def main():
	ureg = pint.UnitRegistry()

	def volume(radius, height):
		return math.pi * radius**2 * height

	vol = volume(3, 5) * ureg('cm^3')

	print('vol [cm^3] =', vol)
	print('vol [cubic inches] =', vol.to('cubic inches'))
	print('vol [gallons] =', vol.to('gallons').m)  # Magnitude.

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
