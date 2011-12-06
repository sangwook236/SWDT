#include "stdafx.h"
#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/cartesian2d.hpp>
#include <boost/geometry/geometries/adapted/tuple_cartesian.hpp>
#include <boost/geometry/geometries/adapted/c_array_cartesian.hpp>
#include <boost/geometry/geometries/adapted/std_as_linestring.hpp>
#include <boost/geometry/multi/multi.hpp>
#include <iostream>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

namespace {

void point2()
{
	// GGL contains several point types:
	// 1: it's own generic type
	boost::geometry::point<double, 2, boost::geometry::cs::cartesian> pt1;

	// 2: it's own type targetted to Cartesian (x,y) coordinates
	boost::geometry::point_2d pt2;

	// 3: it supports Boost tuple's (by including the headerfile)
	boost::tuple<double, double> pt3;

	// 4: it supports normal arrays
	double pt4[2];

	// 5: there are more variants, and you can create your own.
	//    (see therefore the custom_point example)

	// All these types are handled the same way. We show here
	// assigning them and calculating distances.
	assign(pt1, 1, 1);
	assign(pt2, 2, 2);
	boost::geometry::assign(pt3, 3, 3);
	boost::geometry::assign(pt4, 4, 4);

	double d1 = boost::geometry::distance(pt1, pt2);
	double d2 = boost::geometry::distance(pt3, pt4);
	std::cout << "Distances: " << d1 << " and " << d2 << std::endl;

	// (in case you didn't note, distances can be calculated
	//  from points with different point-types)


	// Several ways of construction and setting point values
	// 1: default, empty constructor, causing no initialization at all
	boost::geometry::point_2d p1;

	// 2: as shown above, assign
	boost::geometry::point_2d p2;
	assign(p2, 1, 1);

	// 3: using "set" function
	//    set uses the concepts behind, such that it can be applied for
	//    every point-type (like assign)
	boost::geometry::point_2d p3;
	boost::geometry::set<0>(p3, 1);
	boost::geometry::set<1>(p3, 1);
	// set<2>(p3, 1); //will result in compile-error


	// 4: for any point type, and other geometry objects:
	//    there is the "make" object generator
	//    (this one requires to specify the point-type).
	boost::geometry::point_2d p4 = boost::geometry::make<boost::geometry::point_2d>(1,1);

	// 5: for the point_2d type only: constructor with two values
	boost::geometry::point_2d p5(1,1);

	// 6: for boost tuples you can of course use make_tuple


	// Some ways of getting point values

	// 1: using the "get" function following the concepts behind
	std::cout << boost::geometry::get<0>(p2) << "," << boost::geometry::get<1>(p2) << std::endl;

	// 2: for point-2d only
	std::cout << p2.x() << "," << p2.y() << std::endl;

	// 3: using boost-tuples you of course can boost-tuple-methods
	std::cout << pt3.get<0>() << "," << pt3.get<1>() << std::endl;

	// 4: GGL supports various output formats, e.g. DSV
	//    (delimiter separated values)
	std::cout << boost::geometry::dsv(pt3) << std::endl;


	// Other examples show other types of points, geometries and more algorithms
}

template<typename P>
inline void translate_function(P &p)
{
	p.x(p.x() + 100.0);
}

template<typename P>
struct scale_functor
{
	inline void operator()(P &p)
	{
		p.x(p.x() * 1000.0);
		p.y(p.y() * 1000.0);
	}
};

void linestring2()
{
	// Define a linestring, which is a vector of points, and add some points
	// (we add them deliberately in different ways)
	boost::geometry::linestring_2d ls;

	// points can be created using "make" and added to a linestring using the std:: "push_back"
	ls.push_back(boost::geometry::make<boost::geometry::point_2d>(1.1, 1.1));

	// points can also be assigned using "assign" and added to a linestring using "append"
	boost::geometry::point_2d lp;
	boost::geometry::assign(lp, 2.5, 2.1);
	append(ls, lp);

	// Lines can be streamed using DSV (delimiter separated values)
	std::cout << boost::geometry::dsv(ls) << std::endl;

	// The bounding box of linestrings can be calculated
	boost::geometry::box_2d b;
	envelope(ls, b);
	std::cout << boost::geometry::dsv(b) << std::endl;

	// The length of the line can be calulated
	std::cout << "length: " << length(ls) << std::endl;

	// All things from std::vector can be called, because a linestring is a vector
	std::cout << "number of points 1: " << ls.size() << std::endl;

	// All things from boost ranges can be called because a linestring is considered as a range
	std::cout << "number of points 2: " << boost::size(ls) << std::endl;

	// Generic function from geometry/OGC delivers the same value
	std::cout << "number of points 3: " << num_points(ls) << std::endl;

	// The distance from a point to a linestring can be calculated
	boost::geometry::point_2d p(1.9, 1.2);
	std::cout << "distance of " << boost::geometry::dsv(p) << " to line: " << distance(p, ls) << std::endl;

	// A linestring is a vector. However, some algorithms consider "segments",
	// which are the line pieces between two points of a linestring.
	double d = distance(p, boost::geometry::segment<boost::geometry::point_2d>(ls.front(), ls.back()));
	std::cout << "distance: " << d << std::endl;

	// Add some three points more, let's do it using a classic array.
	// (See documentation for picture of this linestring)
	const double c[][2] = { {3.1, 3.1}, {4.9, 1.1}, {3.1, 1.9} };
	append(ls, c);
	std::cout << "appended: " << boost::geometry::dsv(ls) << std::endl;

	// Output as iterator-pair on a vector
	{
		std::vector<boost::geometry::point_2d> v;
		std::copy(ls.begin(), ls.end(), std::back_inserter(v));

		std::cout << "as vector: " << boost::geometry::dsv(v) << std::endl;

		std::cout << "as it-pair: " << boost::geometry::dsv(std::make_pair(v.begin(), v.end())) << std::endl;
	}

	// All algorithms from std can be used: a linestring is a vector
	std::reverse(ls.begin(), ls.end());
	std::cout << "reversed: " << boost::geometry::dsv(ls) << std::endl;
	std::reverse(boost::begin(ls), boost::end(ls));

	// The other way, using a vector instead of a linestring, is also possible
	std::vector<boost::geometry::point_2d> pv(ls.begin(), ls.end());
	std::cout << "length: " << length(pv) << std::endl;

	// If there are double points in the line, you can use unique to remove them
	// So we add the last point, print, make a unique copy and print
	{
		// (sidenote, we have to make copies, because
		// ls.push_back(ls.back()) often succeeds but
		// IS dangerous and erroneous!
		boost::geometry::point_2d last = ls.back(), first = ls.front();
		ls.push_back(last);
		ls.insert(ls.begin(), first);
	}
	std::cout << "extra duplicate points: " << boost::geometry::dsv(ls) << std::endl;

	{
		boost::geometry::linestring_2d ls_copy;
		std::unique_copy(ls.begin(), ls.end(), std::back_inserter(ls_copy), boost::geometry::equal_to<boost::geometry::point_2d>());
		ls = ls_copy;
		std::cout << "uniquecopy: " << boost::geometry::dsv(ls) << std::endl;
	}

	// Lines can be simplified. This removes points, but preserves the shape
	boost::geometry::linestring_2d ls_simplified;
	simplify(ls, ls_simplified, 0.5);
	std::cout << "simplified: " << boost::geometry::dsv(ls_simplified) << std::endl;


	// for_each:
	// 1) Lines can be visited with std::for_each
	// 2) for_each_point is also defined for all geometries
	// 3) for_each_segment is defined for all geometries to all segments
	// 4) loop is defined for geometries to visit segments
	//    with state apart, and to be able to break out (not shown here)
	{
		boost::geometry::linestring_2d lscopy = ls;
		std::for_each(lscopy.begin(), lscopy.end(), translate_function<boost::geometry::point_2d>);
		for_each_point(lscopy, scale_functor<boost::geometry::point_2d>());
		for_each_point(lscopy, translate_function<boost::geometry::point_2d>);
		std::cout << "modified line: " << boost::geometry::dsv(lscopy) << std::endl;
	}

	// Lines can be clipped using a clipping box. Clipped lines are added to the output iterator
	boost::geometry::box_2d cb(boost::geometry::point_2d(1.5, 1.5), boost::geometry::point_2d(4.5, 2.5));

	std::vector<boost::geometry::linestring_2d> clipped;
	boost::geometry::intersection_inserter<boost::geometry::linestring_2d>(cb, ls, std::back_inserter(clipped));

	// Also possible: clip-output to a vector of vectors
	std::vector<std::vector<boost::geometry::point_2d> > vector_out;
	boost::geometry::intersection_inserter<std::vector<boost::geometry::point_2d> >(cb, ls, std::back_inserter(vector_out));

	std::cout << "clipped output as vector:" << std::endl;
	for (std::vector<std::vector<boost::geometry::point_2d> >::const_iterator it = vector_out.begin(); it != vector_out.end(); ++it)
	{
		std::cout << boost::geometry::dsv(*it) << std::endl;
	}

	// Calculate the convex hull of the linestring
	boost::geometry::polygon_2d hull;
	convex_hull(ls, hull);
	std::cout << "Convex hull:" << boost::geometry::dsv(hull) << std::endl;

	// With DSV you can also use other delimiters, e.g. JSON style
	std::cout << "JSON: " << boost::geometry::dsv(ls, ", ", "[", "]", ", ", "[ ", " ]") << std::endl;
}

void linestring3()
{
	// All the above assumed 2D Cartesian linestrings. 3D is possible as well
	// Let's define a 3D point ourselves, this time using 'float'
	typedef boost::geometry::point<float, 3, boost::geometry::cs::cartesian> point_type;
	typedef boost::geometry::linestring<point_type> line_type;
	line_type line3d;
	line3d.push_back(boost::geometry::make<point_type>(1,2,3));
	line3d.push_back(boost::geometry::make<point_type>(4,5,6));
	line3d.push_back(boost::geometry::make<point_type>(7,8,9));

	// Not all algorithms work on 3d lines. For example convex hull does NOT.
	// But, for example, length, distance, simplify, envelope and stream do.
	std::cout << "3D: length: " << length(line3d) << " line: " << boost::geometry::dsv(line3d) << std::endl;
}

void polygon2()
{
	// Define a polygon and fill the outer ring.
	// In most cases you will read it from a file or database
	boost::geometry::polygon_2d poly;
	{
		const double coor[][2] = {
			{2.0, 1.3}, {2.4, 1.7}, {2.8, 1.8}, {3.4, 1.2}, {3.7, 1.6},
			{3.4, 2.0}, {4.1, 3.0}, {5.3, 2.6}, {5.4, 1.2}, {4.9, 0.8}, {2.9, 0.7},
			{2.0, 1.3} // closing point is opening point
		};
		assign(poly, coor);
	}

	// Polygons should be closed, and directed clockwise. If you're not sure if that is the case,
	// call the correct algorithm
	correct(poly);

	// Polygons can be streamed as text
	// (or more precisely: as DSV (delimiter separated values))
	std::cout << boost::geometry::dsv(poly) << std::endl;

	// As with lines, bounding box of polygons can be calculated
	boost::geometry::box_2d b;
	envelope(poly, b);
	std::cout << boost::geometry::dsv(b) << std::endl;

	// The area of the polygon can be calulated
	std::cout << "area: " << area(poly) << std::endl;

	// And the centroid, which is the center of gravity
	boost::geometry::point_2d cent;
	centroid(poly, cent);
	std::cout << "centroid: " << boost::geometry::dsv(cent) << std::endl;


	// The number of points have to called per ring separately
	std::cout << "number of points in outer ring: " << poly.outer().size() << std::endl;

	// Polygons can have one or more inner rings, also called holes, donuts, islands, interior rings.
	// Let's add one
	{
		poly.inners().resize(1);
		boost::geometry::linear_ring<boost::geometry::point_2d> &inner = poly.inners().back();

		const double coor[][2] = { {4.0, 2.0}, {4.2, 1.4}, {4.8, 1.9}, {4.4, 2.2}, {4.0, 2.0} };
		assign(inner, coor);
	}

	correct(poly);

	std::cout << "with inner ring:" << boost::geometry::dsv(poly) << std::endl;
	// The area of the polygon is changed of course
	std::cout << "new area of polygon: " << area(poly) << std::endl;
	centroid(poly, cent);
	std::cout << "new centroid: " << boost::geometry::dsv(cent) << std::endl;

	// You can test whether points are within a polygon
	std::cout << "point in polygon:"
		<< " p1: "  << std::boolalpha << within(boost::geometry::make<boost::geometry::point_2d>(3.0, 2.0), poly)
		<< " p2: "  << within(boost::geometry::make<boost::geometry::point_2d>(3.7, 2.0), poly)
		<< " p3: "  << within(boost::geometry::make<boost::geometry::point_2d>(4.4, 2.0), poly)
		<< std::endl;

	// As with linestrings and points, you can derive from polygon to add, for example,
	// fill color and stroke color. Or SRID (spatial reference ID). Or Z-value. Or a property map.
	// We don't show this here.

	// Clip the polygon using a bounding box
	boost::geometry::box_2d cb(boost::geometry::make<boost::geometry::point_2d>(1.5, 1.5), boost::geometry::make<boost::geometry::point_2d>(4.5, 2.5));
	typedef std::vector<boost::geometry::polygon_2d> polygon_list;
	polygon_list v;

	boost::geometry::intersection_inserter<boost::geometry::polygon_2d>(cb, poly, std::back_inserter(v));
	std::cout << "Clipped output polygons" << std::endl;
	for (polygon_list::const_iterator it = v.begin(); it != v.end(); ++it)
	{
		std::cout << boost::geometry::dsv(*it) << std::endl;
	}

	typedef boost::geometry::multi_polygon<boost::geometry::polygon_2d> polygon_set;
	polygon_set ps;
	boost::geometry::union_inserter<boost::geometry::polygon_2d>(cb, poly, std::back_inserter(ps));

	boost::geometry::polygon_2d hull;
	convex_hull(poly, hull);
	std::cout << "Convex hull:" << boost::geometry::dsv(hull) << std::endl;

	// If you really want:
	//   You don't have to use a vector, you can define a polygon with a deque
	//   You can specify the container for the points and for the inner rings independantly

	typedef boost::geometry::polygon<boost::geometry::point_2d, std::vector, std::deque> polygon_type;
	polygon_type poly2;
	boost::geometry::ring_type<polygon_type>::type &ring = exterior_ring(poly2);
	append(ring, boost::geometry::make<boost::geometry::point_2d>(2.8, 1.9));
	append(ring, boost::geometry::make<boost::geometry::point_2d>(2.9, 2.4));
	append(ring, boost::geometry::make<boost::geometry::point_2d>(3.3, 2.2));
	append(ring, boost::geometry::make<boost::geometry::point_2d>(3.2, 1.8));
	append(ring, boost::geometry::make<boost::geometry::point_2d>(2.8, 1.9));
	std::cout << boost::geometry::dsv(poly2) << std::endl;
}

void polygon3()
{
	// All the above assumed 2D Cartesian linestrings. 3D is possible as well
	// Let's define a 3D point ourselves, this time using 'float'
	typedef boost::geometry::point<float, 3, boost::geometry::cs::cartesian> point_type;
	typedef boost::geometry::polygon<point_type> polygon_type;
	polygon_type poly3d;
#if 0
	boost::geometry::ring_type<polygon_type>::type &ring = exterior_ring(poly3d);
	append(ring, boost::geometry::make<point_type>(2,2,3));
	append(ring, boost::geometry::make<point_type>(2,5,6));
	append(ring, boost::geometry::make<point_type>(2,8,1));
	append(ring, boost::geometry::make<point_type>(2,2,3));
#else
	const double coor[][3] = { {2,2,3}, {2,5,6}, {2,8,1}, {2,2,3} };
	assign(poly3d, coor);
#endif

	std::cout << "3D: polygon: " << boost::geometry::dsv(poly3d) << std::endl;

	std::cout << "3D: length: " << length(poly3d) << std::endl;  // Oops !!!: error
	//std::cout << "3D: area: " << area(poly3d) << std::endl;
	//point_type cent;
	//centroid(poly3d, cent);
	//std::cout << "3D: centroid: " << boost::geometry::dsv(cent) << std::endl;

	std::cout << "point in polygon:" << std::boolalpha << within(boost::geometry::make<point_type>(2, 4, 3), poly3d) << std::endl;
}

void transform2()
{
    boost::geometry::point_2d p(1, 1);
    boost::geometry::point_2d p2;

    // Example: translate a point over (5,5)
    boost::geometry::strategy::transform::translate_transformer<boost::geometry::point_2d, boost::geometry::point_2d> translate(5, 5);

    transform(p, p2, translate);
    std::cout << "transformed point " << boost::geometry::dsv(p2) << std::endl;

    // Transform a polygon
    boost::geometry::polygon_2d poly, poly2;
    const double coor[][2] = { {0, 0}, {0, 7}, {2, 2}, {2, 0}, {0, 0} };
    // note that for this syntax you have to include the two
    // include files above (c_array_cartesian.hpp, std_as_linestring.hpp)
    assign(poly, coor);
    //read_wkt("POLYGON((0 0,0 7,4 2,2 0,0 0))", poly);
    transform(poly, poly2, translate);

    std::cout << "source      polygon " << boost::geometry::dsv(poly) << std::endl;
    std::cout << "transformed polygon " << boost::geometry::dsv(poly2) << std::endl;

    // Many more transformations are possible:
    // - from Cartesian to Spherical coordinate systems and back
    // - from Cartesian to Cartesian (mapping, affine transformations) and back (inverse)
    // - Map Projections
    // - from Degree to Radian and back in spherical or geographic coordinate systems
}

void graph_route()
{
	// ref: example
}

}  // unnamed namespace

void geometry()
{
	point2();
	linestring2();
	linestring3();
	polygon2();
	polygon3();

	transform2();

	graph_route();
}
