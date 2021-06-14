#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/Toblerity/Shapely

import numpy as np
from shapely.geometry import Point, MultiPoint, LineString
from shapely.geometry import asPoint, asLineString, asMultiPoint
from shapely.geometry.polygon import Polygon, LinearRing
import shapely.affinity
import shapely.ops

# REF [site] >> https://shapely.readthedocs.io/en/latest/manual.html
def operation_example():
	patch = Point(0.0, 0.0).buffer(10.0)
	print('patch = {}.'.format(patch))
	print('patch.area = {}.'.format(patch.area))

	line = LineString([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])
	dilated = line.buffer(0.5)
	eroded = dilated.buffer(-0.3)

	ring = LinearRing([(0, 0), (1, 1), (1, 0)])

	point = Point(0.5, 0.5)
	polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
	print('polygon.contains(point) = {}.'.format(polygon.contains(point)))

	#box = shapely.geometry.box(minx, miny, maxx, maxy, ccw=True)

	pa = asPoint(np.array([0.0, 0.0]))
	la = asLineString(np.array([[1.0, 2.0], [3.0, 4.0]]))
	ma = asMultiPoint(np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]))

	data = {'type': 'Point', 'coordinates': (0.0, 0.0)}
	geom = shapely.geometry.shape(data)
	#geom = shapely.geometry.asShape(data)

	#--------------------
	#geom = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
	geom = LineString([(0, 0), (10, 0), (10, 5), (20, 5)])

	print('polygon.has_z = {}.'.format(geom.has_z))
	if isinstance(geom, LinearRing):
		print('polygon.is_ccw = {}.'.format(geom.is_ccw))  # LinearRing only.
	print('polygon.is_empty = {}.'.format(geom.is_empty))
	print('polygon.is_ring = {}.'.format(geom.is_ring))
	print('polygon.is_simple = {}.'.format(geom.is_simple))
	print('polygon.is_valid = {}.'.format(geom.is_valid))

	print('polygon.coords = {}.'.format(geom.coords))

	print('polygon.area = {}.'.format(geom.area))
	print('polygon.bounds = {}.'.format(geom.bounds))
	print('polygon.length = {}.'.format(geom.length))
	print('polygon.minimum_clearance = {}.'.format(geom.minimum_clearance))
	print('polygon.geom_type = {}.'.format(geom.geom_type))

	print('polygon.boundary = {}.'.format(geom.boundary))
	print('polygon.centroid = {}.'.format(geom.centroid))

	print('polygon.convex_hull.exterior.coords.xy = {}.'.format(list((x, y) for x, y in zip(*geom.convex_hull.exterior.coords.xy))))
	print('polygon.envelope = {}.'.format(geom.envelope))
	print('polygon.minimum_rotated_rectangle = {}.'.format(geom.minimum_rotated_rectangle))

	#polygon.parallel_offset(distance, side, resolution=16, join_style=1, mitre_limit=5.0)
	#polygon.simplify(tolerance, preserve_topology=True)

	#--------------------
	poly1 = Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])
	poly2 = Polygon([(1, 1), (4, 1), (4, 3.5), (1, 3.5)])

	print('poly1.intersection(poly2) = {}.'.format(poly1.intersection(poly2)))
	print('poly1.difference(poly2) = {}.'.format(poly1.difference(poly2)))
	print('poly1.symmetric_difference(poly2) = {}.'.format(poly1.symmetric_difference(poly2)))
	print('poly1.union(poly2) = {}.'.format(poly1.union(poly2)))

	print('poly1.equals(poly2) = {}.'.format(poly1.equals(poly2)))
	print('poly1.almost_equals(poly2, decimal=6) = {}.'.format(poly1.almost_equals(poly2, decimal=6)))
	print('poly1.contains(poly2) = {}.'.format(poly1.contains(poly2)))
	print('poly1.covers(poly2) = {}.'.format(poly1.covers(poly2)))
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
	print('The Hausdorff distance between two convex polygons = {}.'.format(poly1.hausdorff_distance(poly2)))
	print('The representative point of a polygon = {}.'.format(poly1.representative_point()))

	#--------------------
	lines = [
		((0, 0), (1, 1)),
		((0, 0), (0, 1)),
		((0, 1), (1, 1)),
		((1, 1), (1, 0)),
		((1, 0), (0, 0)),
		((1, 1), (2, 0)),
		((2, 0), (1, 0)),
		((5, 5), (6, 6)),
		((1, 1), (100, 100)),
	]

	polys = shapely.ops.polygonize(lines)
	result, dangles, cuts, invalids = shapely.ops.polygonize_full(lines)
	merged_line = shapely.ops.linemerge(lines)

	# Efficient unions.
	polygons = [Point(i, 0).buffer(0.7) for i in range(5)]
	shapely.ops.unary_union(polygons)
	shapely.ops.cascaded_union(polygons)

	# Delaunay triangulation.
	points = MultiPoint([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])
	triangles = shapely.ops.triangulate(points)

	# Voronoi diagram.
	#points = MultiPoint([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)])
	#regions = shapely.ops.voronoi_diagram(points)

	# Nearest points.
	triangle = Polygon([(0, 0), (1, 0), (0.5, 1), (0, 0)])
	square = Polygon([(0, 2), (1, 2), (1, 3), (0, 3), (0, 2)])
	nearest_points = shapely.ops.nearest_points(triangle, square)

	square = Polygon([(1,1), (2, 1), (2, 2), (1, 2), (1, 1)])
	line = LineString([(0,0), (0.8, 0.8), (1.8, 0.95), (2.6, 0.5)])
	result = shapely.ops.snap(line, square, 0.5)

	g1 = LineString([(0, 0), (10, 0), (10, 5), (20, 5)])
	g2 = LineString([(5, 0), (30, 0), (30, 5), (0, 5)])
	forward, backward = shapely.ops.shared_paths(g1, g2)

	pt = Point((1, 1))
	line = LineString([(0, 0), (2, 2)])
	result = shapely.ops.split(line, pt)

	ls = LineString((i, 0) for i in range(6))
	shapely.ops.substring(ls, start_dist=1, end_dist=3)
	shapely.ops.substring(ls, start_dist=3, end_dist=1)
	shapely.ops.substring(ls, start_dist=1, end_dist=-3)
	shapely.ops.substring(ls, start_dist=0.2, end_dist=-0.6, normalized=True)

# REF [site] >> https://shapely.readthedocs.io/en/latest/manual.html
def transformation_example():
	line = LineString([(1, 3), (1, 1), (4, 1)])
	translated_line = shapely.affinity.translate(line, xoff=10.0, yoff=10.0, zoff=10.0)
	rotated_a = shapely.affinity.rotate(line, 90)
	rotated_b = shapely.affinity.rotate(line, 90, origin='centroid')

	triangle = Polygon([(1, 1), (2, 3), (3, 1)])
	triangle_a = shapely.affinity.scale(triangle, xfact=1.5, yfact=-1)
	triangle_a.exterior.coords[:]
	triangle_b = shapely.affinity.scale(triangle, xfact=2, origin=(1, 1))
	triangle_b.exterior.coords[:]

	#matrix = [a, b, d, e, xoff, yoff]  # For 2D.
	#matrix = [a, b, c, d, e, f, g, h, i, xoff, yoff, zoff]  # For 3D.
	#shapely.affinity.affine_transform(line, matrix)

	#shapely.ops.transform(func, geom)

def main():
	operation_example()
	transformation_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
