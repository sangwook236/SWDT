#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# REF [site] >> https://github.com/Toblerity/Shapely

def simple_example():
	patch = Point(0.0, 0.0).buffer(10.0)
	print('patch =', patch)
	print('patch.area =', patch.area)

	point = Point(0.5, 0.5)
	polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
	print('polygon.contains(point) =', polygon.contains(point))

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
