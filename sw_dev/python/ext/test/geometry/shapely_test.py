#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/Toblerity/Shapely

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# REF [site] >> https://shapely.readthedocs.io/en/latest/manual.html
def simple_example():
	patch = Point(0.0, 0.0).buffer(10.0)
	print('patch = {}.'.format(patch))
	print('patch.area = {}.'.format(patch.area))

	#--------------------
	point = Point(0.5, 0.5)
	polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
	print('polygon.contains(point) = {}.'.format(polygon.contains(point)))

	print('polygon.has_z = {}.'.format(polygon.has_z))
	print('polygon.is_ccw = {}.'.format(polygon.is_ccw))
	print('polygon.is_empty = {}.'.format(polygon.is_empty))
	print('polygon.is_ring = {}.'.format(polygon.is_ring))
	print('polygon.is_valid = {}.'.format(polygon.is_valid))

	poly1 = Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])
	poly2 = Polygon([(1, 1), (4, 1), (4, 3.5), (1, 3.5)])
	intersection = poly1.intersection(poly2)
	print('The intersection area of two polygons = {}.'.format(intersection.area))

	print('poly1.equals(poly2) = {}.'.format(poly1.equals(poly2)))
	print('poly1.almost_equals(poly2, decimal=6) = {}.'.format(poly1.almost_equals(poly2, decimal=6)))
	print('poly1.contains(poly2) = {}.'.format(poly1.contains(poly2)))
	print('poly1.crosses(poly2) = {}.'.format(poly1.crosses(poly2)))
	print('poly1.disjoint(poly2) = {}.'.format(poly1.disjoint(poly2)))
	print('poly1.intersects(poly2) = {}.'.format(poly1.intersects(poly2)))
	print('poly1.overlaps(poly2) = {}.'.format(poly1.overlaps(poly2)))
	print('poly1.touches(poly2) = {}.'.format(poly1.touches(poly2)))
	print('poly1.within(poly2) = {}.'.format(poly1.within(poly2)))

	#--------------------
	poly1 = Polygon([(40, 50), (60, 50), (60, 90), (40, 90)])
	poly2 = Polygon([(10, 100), (100, 100), (100, 150), (10, 150)])
	print('The distance between two convex polygons = {}.'.format(poly1.distance(poly2)))

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
