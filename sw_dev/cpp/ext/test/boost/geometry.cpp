#include <boost/foreach.hpp>
#include <boost/random.hpp>
#include <boost/bind.hpp>
#include <boost/range.hpp>
#include <boost/static_assert.hpp>

#include <sstream>
#include <fstream>
#include <iostream>

#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/multi/multi.hpp>
#include <boost/geometry/multi/geometries/multi_polygon.hpp>
#include <boost/geometry/algorithms/centroid.hpp>
#include <boost/geometry/strategies/transform.hpp>
#include <boost/geometry/strategies/transform/matrix_transformers.hpp>

// Optional includes and defines to handle c-arrays as points, std::vectors as linestrings.
#include <boost/geometry/geometries/register/linestring.hpp>
#include <boost/geometry/geometries/adapted/boost_tuple.hpp>
#include <boost/geometry/geometries/adapted/boost_array.hpp>
#include <boost/geometry/geometries/adapted/boost_polygon/point.hpp>
#include <boost/geometry/geometries/adapted/c_array.hpp>

//#include <boost/geometry/domains/gis/io/wkt/read_wkt.hpp>
#include <boost/geometry/io/wkt/wkt.hpp>

#include <boost/mpl/int.hpp>
#include <boost/geometry/core/cs.hpp>
#include <boost/geometry/geometries/point.hpp>

//#define HAVE_SVG
#if defined(HAVE_SVG)
#  include <boost/geometry/extensions/io/svg/write_svg.hpp>
#  include <boost/geometry/extensions/io/svg/svg_mapper.hpp>
#endif


BOOST_GEOMETRY_REGISTER_C_ARRAY_CS(cs::cartesian)
BOOST_GEOMETRY_REGISTER_BOOST_ARRAY_CS(cs::cartesian)
BOOST_GEOMETRY_REGISTER_BOOST_TUPLE_CS(cs::cartesian)

BOOST_GEOMETRY_REGISTER_LINESTRING_TEMPLATED(std::vector)
BOOST_GEOMETRY_REGISTER_LINESTRING_TEMPLATED(std::deque)


namespace boost { namespace geometry
{

namespace model { namespace d3
{

template<typename CoordinateType, typename CoordinateSystem = boost::geometry::cs::cartesian>
class point_xyz : public boost::geometry::model::point<CoordinateType, 3, CoordinateSystem>
{
public:

    /// Default constructor, does not initialize anything
    inline point_xyz()
    : boost::geometry::model::point<CoordinateType, 3, CoordinateSystem>()
    {}

    /// Constructor with x/y values
    inline point_xyz(CoordinateType const &x, CoordinateType const &y)
    : boost::geometry::model::point<CoordinateType, 3, CoordinateSystem>(x, y)
    {}

    /// Get x-value.
    inline CoordinateType const & x() const
    { return this->template get<0>(); }

    /// Get y-value.
    inline CoordinateType const & y() const
    { return this->template get<1>(); }

    /// Get z-value.
    inline CoordinateType const & z() const
    { return this->template get<2>(); }

    /// Set x-value.
    inline void x(CoordinateType const &v)
    { this->template set<0>(v); }

    /// Set y-value.
    inline void y(CoordinateType const &v)
    { this->template set<1>(v); }

    /// Set z-value.
    inline void z(CoordinateType const &v)
    { this->template set<2>(v); }
};

}  // namespace d2
}  // namespace model

// Adapt the point_xyz to the concept.
#ifndef DOXYGEN_NO_TRAITS_SPECIALIZATIONS
namespace traits
{

template <typename CoordinateType, typename CoordinateSystem>
struct tag<boost::geometry::model::d3::point_xyz<CoordinateType, CoordinateSystem> >
{
    typedef point_tag type;
};

template<typename CoordinateType, typename CoordinateSystem>
struct coordinate_type<boost::geometry::model::d3::point_xyz<CoordinateType, CoordinateSystem> >
{
    typedef CoordinateType type;
};

template<typename CoordinateType, typename CoordinateSystem>
struct coordinate_system<boost::geometry::model::d3::point_xyz<CoordinateType, CoordinateSystem> >
{
    typedef CoordinateSystem type;
};

template<typename CoordinateType, typename CoordinateSystem>
struct dimension<boost::geometry::model::d3::point_xyz<CoordinateType, CoordinateSystem> >
    : boost::mpl::int_<3>
{};

template<typename CoordinateType, typename CoordinateSystem, std::size_t Dimension>
struct access<model::d3::point_xyz<CoordinateType, CoordinateSystem>, Dimension>
{
    static inline CoordinateType get(
        model::d3::point_xyz<CoordinateType, CoordinateSystem> const &p)
    {
        return p.template get<Dimension>();
    }

    static inline void set(model::d3::point_xyz<CoordinateType, CoordinateSystem> &p,
        CoordinateType const &value)
    {
        p.template set<Dimension>(value);
    }
};

}  // namespace traits
#endif  // DOXYGEN_NO_TRAITS_SPECIALIZATIONS

}  // namespace geometry
}  // namespace boost

namespace {
namespace local {

void point_2d()
{
	// GGL contains several point types:
	// 1: it's own generic type.
	boost::geometry::model::point<double, 2, boost::geometry::cs::cartesian> pt1;

	// 2: it's own type targetted to Cartesian (x,y) coordinates.
	boost::geometry::model::d2::point_xy<double> pt2;

	// 3: it supports Boost tuple's (by including the headerfile).
	boost::tuple<double, double> pt3;

	// 4: it supports normal arrays.
	double pt4[2];

    // 5: it supports arrays-as-points from Boost.Array.
    boost::array<double, 2> pt5;

    // 6: it supports points from Boost.Polygon
    boost::polygon::point_data<double> pt6;

    // 7: in the past there was a typedef point_2d.
    //    But users are now supposted to do that themselves:
    typedef boost::geometry::model::d2::point_xy<double> point_2d_t;
    point_2d_t pt7;

	// All these types are handled the same way. We show here assigning them and calculating distances.
	boost::geometry::assign_values(pt1, 1, 1);
	boost::geometry::assign_values(pt2, 2, 2);
	boost::geometry::assign_values(pt3, 3, 3);
	boost::geometry::assign_values(pt4, 4, 4);
	boost::geometry::assign_values(pt5, 5, 5);
	boost::geometry::assign_values(pt6, 6, 6);
	boost::geometry::assign_values(pt7, 7, 7);

	const double d1 = boost::geometry::distance(pt1, pt2);
	const double d2 = boost::geometry::distance(pt3, pt4);
    const double d3 = boost::geometry::distance(pt5, pt6);
	std::cout << "Distances: " << d1 << " and " << d2 << " and " << d3 << std::endl;

	// (in case you didn't note, distances can be calculated from points with different point-types).

	// Several ways of construction and setting point values.
	// 1: default, empty constructor, causing no initialization at all.
	boost::geometry::model::d2::point_xy<double> p1;

	// 2: as shown above, assign.
	boost::geometry::model::d2::point_xy<double> p2;
	boost::geometry::assign_values(p2, 1, 1);

	// 3: using "set" function.
	//    Set uses the concepts behind, such that it can be applied for every point-type (like assign).
	boost::geometry::model::d2::point_xy<double> p3;
	boost::geometry::set<0>(p3, 1.0);
	boost::geometry::set<1>(p3, 1);
	// boost::geometry::set<2>(p3, 1);  // Will result in compile-error.


	// 4: for any point type, and other geometry objects:
	//    there is the "make" object generator.
	//    (this one requires to specify the point-type).
	boost::geometry::model::d2::point_xy<double> p4 = boost::geometry::make<boost::geometry::model::d2::point_xy<double> >(1, 1);
	//boost::geometry::model::d2::point_xy<double> p4 = boost::geometry::make<boost::geometry::model::d2::point_xy<double>, double>(1.0, 1.0);

	// 5: for the point_2d type only: constructor with two values.
	boost::geometry::model::d2::point_xy<double> p5(1, 1);

	// 6: for boost tuples you can of course use make_tuple.


	// Some ways of getting point values.

	// 1: using the "get" function following the concepts behind.
	std::cout << boost::geometry::get<0>(p2) << "," << boost::geometry::get<1>(p2) << std::endl;

	// 2: for point-2d only.
	std::cout << p2.x() << "," << p2.y() << std::endl;

	// 3: using boost-tuples you of course can boost-tuple-methods.
	std::cout << pt3.get<0>() << "," << pt3.get<1>() << std::endl;

	// 4: GGL supports various output formats, e.g. DSV (delimiter separated values).
	std::cout << boost::geometry::dsv(pt3) << std::endl;


	// Other examples show other types of points, geometries and more algorithms.
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

void linestring_2d()
{
    typedef boost::geometry::model::d2::point_xy<double> point_2d_t;
    typedef boost::geometry::model::linestring<point_2d_t> linestring_2d_t;
    typedef boost::geometry::model::box<point_2d_t> box_2d_t;

	// Define a linestring, which is a vector of points, and add some points
	// (we add them deliberately in different ways).
	linestring_2d_t ls;

	// Points can be created using "make" and added to a linestring using the std:: "push_back".
	ls.push_back(boost::geometry::make<point_2d_t>(1.1, 1.1));

	// Points can also be assigned using "assign" and added to a linestring using "append".
	point_2d_t lp;
	boost::geometry::assign_values(lp, 2.5, 2.1);
	boost::geometry::append(ls, lp);

	// Lines can be streamed using DSV (delimiter separated values)
	std::cout << boost::geometry::dsv(ls) << std::endl;

	// The bounding box of linestrings can be calculated.
	box_2d_t b;
	boost::geometry::envelope(ls, b);
	std::cout << boost::geometry::dsv(b) << std::endl;

	// The length of the line can be calulated.
	std::cout << "length: " << boost::geometry::length(ls) << std::endl;

	// All things from std::vector can be called, because a linestring is a vector.
	std::cout << "number of points 1: " << ls.size() << std::endl;

	// All things from boost ranges can be called because a linestring is considered as a range.
	std::cout << "number of points 2: " << boost::size(ls) << std::endl;

	// Generic function from geometry/OGC delivers the same value.
	std::cout << "number of points 3: " << boost::geometry::num_points(ls) << std::endl;

	// The distance from a point to a linestring can be calculated.
	point_2d_t p(1.9, 1.2);
	std::cout << "distance of " << boost::geometry::dsv(p) << " to line: " << boost::geometry::distance(p, ls) << std::endl;

	// A linestring is a vector. However, some algorithms consider "segments",
	// which are the line pieces between two points of a linestring.
	const double d = boost::geometry::distance(p, boost::geometry::model::segment<point_2d_t>(ls.front(), ls.back()));
	std::cout << "distance: " << d << std::endl;

	// Add some three points more, let's do it using a classic array.
	// (See documentation for picture of this linestring).
	const double c[][2] = { {3.1, 3.1}, {4.9, 1.1}, {3.1, 1.9} };
	boost::geometry::append(ls, c);
	std::cout << "appended: " << boost::geometry::dsv(ls) << std::endl;

	// Output as iterator-pair on a vector.
	{
		std::vector<point_2d_t> v;
		std::copy(ls.begin(), ls.end(), std::back_inserter(v));

		std::cout << "as vector: " << boost::geometry::dsv(v) << std::endl;
		//std::cout << "as it-pair: " << boost::geometry::dsv(std::make_pair(v.begin(), v.end())) << std::endl;  // Oops !!! compile-time error.
	}

	// All algorithms from std can be used: a linestring is a vector.
	std::reverse(ls.begin(), ls.end());
	std::cout << "reversed: " << boost::geometry::dsv(ls) << std::endl;
	std::reverse(boost::begin(ls), boost::end(ls));

	// The other way, using a vector instead of a linestring, is also possible.
	std::vector<point_2d_t> pv(ls.begin(), ls.end());
	std::cout << "length: " << boost::geometry::length(pv) << std::endl;

	// If there are double points in the line, you can use unique to remove them.
	// So we add the last point, print, make a unique copy and print.
	{
		// (sidenote, we have to make copies, because ls.push_back(ls.back()) often succeeds but IS dangerous and erroneous!
		point_2d_t last = ls.back(), first = ls.front();
		ls.push_back(last);
		ls.insert(ls.begin(), first);
	}
	std::cout << "extra duplicate points: " << boost::geometry::dsv(ls) << std::endl;

	{
		linestring_2d_t ls_copy;
		std::unique_copy(ls.begin(), ls.end(), std::back_inserter(ls_copy), boost::geometry::equal_to<point_2d_t>());
		ls = ls_copy;
		std::cout << "uniquecopy: " << boost::geometry::dsv(ls) << std::endl;
	}

	// Lines can be simplified. This removes points, but preserves the shape.
	linestring_2d_t ls_simplified;
	boost::geometry::simplify(ls, ls_simplified, 0.5);
	std::cout << "simplified: " << boost::geometry::dsv(ls_simplified) << std::endl;

	// for_each:
	// 1) Lines can be visited with std::for_each.
	// 2) for_each_point is also defined for all geometries.
	// 3) for_each_segment is defined for all geometries to all segments.
	// 4) loop is defined for geometries to visit segments
	//    with state apart, and to be able to break out (not shown here).
	{
		linestring_2d_t lscopy = ls;
		std::for_each(lscopy.begin(), lscopy.end(), translate_function<point_2d_t>);
		boost::geometry::for_each_point(lscopy, scale_functor<point_2d_t>());
		boost::geometry::for_each_point(lscopy, translate_function<point_2d_t>);
		std::cout << "modified line: " << boost::geometry::dsv(lscopy) << std::endl;
	}

	// Lines can be clipped using a clipping box. Clipped lines are added to the output iterator.
	box_2d_t cb(point_2d_t(1.5, 1.5), point_2d_t(4.5, 2.5));

	std::vector<linestring_2d_t> clipped;
	boost::geometry::intersection(cb, ls, clipped);

	// Also possible: clip-output to a vector of vectors.
	std::vector<std::vector<point_2d_t> > vector_out;
	boost::geometry::intersection(cb, ls, vector_out);

	std::cout << "clipped output as vector:" << std::endl;
	for (std::vector<std::vector<point_2d_t> >::const_iterator it = vector_out.begin(); it != vector_out.end(); ++it)
	{
		std::cout << boost::geometry::dsv(*it) << std::endl;
	}

	// Calculate the convex hull of the linestring.
	boost::geometry::model::polygon<point_2d_t> hull;
	boost::geometry::convex_hull(ls, hull);
	std::cout << "Convex hull:" << boost::geometry::dsv(hull) << std::endl;


	// With DSV you can also use other delimiters, e.g. JSON style.
	std::cout << "JSON: " << boost::geometry::dsv(ls, ", ", "[", "]", ", ", "[ ", " ]") << std::endl;
}

void linestring_3d()
{
	// All the above assumed 2D Cartesian linestrings. 3D is possible as well.
	// Let's define a 3D point ourselves, this time using 'float'.
	typedef boost::geometry::model::point<float, 3, boost::geometry::cs::cartesian> point_3d_t;
	typedef boost::geometry::model::linestring<point_3d_t> line_3d_t;

	line_3d_t line3;

	line3.push_back(boost::geometry::make<point_3d_t>(1, 2, 3));
	line3.push_back(boost::geometry::make<point_3d_t>(4, 5, 6));
	line3.push_back(boost::geometry::make<point_3d_t>(7, 8, 9));

	// Not all algorithms work on 3d lines. For example convex hull does NOT.
	// But, for example, length, distance, simplify, envelope and stream do.
    std::cout << "3D: length: " << boost::geometry::length(line3) << " line: " << boost::geometry::dsv(line3) << std::endl;
}

void polygon_2d()
{
	typedef boost::geometry::model::d2::point_xy<double> point_2d_t;
	typedef boost::geometry::model::polygon<point_2d_t> polygon_2d_t;
	typedef boost::geometry::model::box<point_2d_t> box_2d_t;

	// Define a polygon and fill the outer ring.
	// In most cases you will read it from a file or database.
	polygon_2d_t poly;
	{
		const double coor[][2] = {
			{2.0, 1.3}, {2.4, 1.7}, {2.8, 1.8}, {3.4, 1.2}, {3.7, 1.6},
			{3.4, 2.0}, {4.1, 3.0}, {5.3, 2.6}, {5.4, 1.2}, {4.9, 0.8}, {2.9, 0.7},
			{2.0, 1.3} // closing point is opening point
		};
		boost::geometry::assign_points(poly, coor);
	}

	// Polygons should be closed, and directed clockwise. If you're not sure if that is the case, call the correct algorithm.
	boost::geometry::correct(poly);

	// Polygons can be streamed as text (or more precisely: as DSV (delimiter separated values)).
	std::cout << boost::geometry::dsv(poly) << std::endl;

	// As with lines, bounding box of polygons can be calculated.
	box_2d_t b;
	boost::geometry::envelope(poly, b);
	std::cout << boost::geometry::dsv(b) << std::endl;

	// The area of the polygon can be calulated
	std::cout << "area: " << boost::geometry::area(poly) << std::endl;

	// And the centroid, which is the center of gravity.
	point_2d_t cent;
	boost::geometry::centroid(poly, cent);
	std::cout << "centroid: " << boost::geometry::dsv(cent) << std::endl;


	// The number of points have to called per ring separately.
	std::cout << "number of points in outer ring: " << poly.outer().size() << std::endl;

	// Polygons can have one or more inner rings, also called holes, donuts, islands, interior rings.
	// Let's add one.
	{
		poly.inners().resize(1);
		boost::geometry::model::ring<point_2d_t> &inner = poly.inners().back();

		const double coor[][2] = { {4.0, 2.0}, {4.2, 1.4}, {4.8, 1.9}, {4.4, 2.2}, {4.0, 2.0} };
		boost::geometry::assign_points(inner, coor);
	}

	boost::geometry::correct(poly);

	std::cout << "with inner ring:" << boost::geometry::dsv(poly) << std::endl;
	// The area of the polygon is changed of course.
	std::cout << "new area of polygon: " << boost::geometry::area(poly) << std::endl;
	boost::geometry::centroid(poly, cent);
	std::cout << "new centroid: " << boost::geometry::dsv(cent) << std::endl;

	// You can test whether points are within a polygon.
	std::cout << "point in polygon:"
		<< " p1: "  << std::boolalpha << boost::geometry::within(boost::geometry::make<point_2d_t>(3.0, 2.0), poly)
		<< " p2: "  << boost::geometry::within(boost::geometry::make<point_2d_t>(3.7, 2.0), poly)
		<< " p3: "  << boost::geometry::within(boost::geometry::make<point_2d_t>(4.4, 2.0), poly)
		<< std::endl;

	// As with linestrings and points, you can derive from polygon to add, for example,
	// fill color and stroke color. Or SRID (spatial reference ID). Or Z-value. Or a property map.
	// We don't show this here.

	// Clip the polygon using a bounding box.
	box_2d_t cb(boost::geometry::make<point_2d_t>(1.5, 1.5), boost::geometry::make<point_2d_t>(4.5, 2.5));
	typedef std::vector<polygon_2d_t> polygon_2d_list_t;

	polygon_2d_list_t v;
	boost::geometry::intersection(cb, poly, v);
	std::cout << "Clipped output polygons" << std::endl;
	for (polygon_2d_list_t::const_iterator it = v.begin(); it != v.end(); ++it)
	{
		std::cout << boost::geometry::dsv(*it) << std::endl;
	}

	typedef boost::geometry::model::multi_polygon<polygon_2d_t> polygon_2d_set_t;

	polygon_2d_set_t ps;
	boost::geometry::union_(cb, poly, ps);

	polygon_2d_t hull;
	boost::geometry::convex_hull(poly, hull);
	std::cout << "Convex hull:" << boost::geometry::dsv(hull) << std::endl;

	// If you really want:
	//   You don't have to use a vector, you can define a polygon with a deque.
	//   You can specify the container for the points and for the inner rings independantly.

	typedef boost::geometry::model::polygon<point_2d_t, true, true, std::deque, std::deque> deque_polygon_t;

	deque_polygon_t poly2;
	boost::geometry::ring_type<deque_polygon_t>::type &ring = boost::geometry::exterior_ring(poly2);
	boost::geometry::append(ring, boost::geometry::make<point_2d_t>(2.8, 1.9));
	boost::geometry::append(ring, boost::geometry::make<point_2d_t>(2.9, 2.4));
	boost::geometry::append(ring, boost::geometry::make<point_2d_t>(3.3, 2.2));
	boost::geometry::append(ring, boost::geometry::make<point_2d_t>(3.2, 1.8));
	boost::geometry::append(ring, boost::geometry::make<point_2d_t>(2.8, 1.9));
	std::cout << boost::geometry::dsv(poly2) << std::endl;
}

void polygon_3d()
{
	// All the above assumed 2D Cartesian linestrings. 3D is possible as well.
	// Let's define a 3D point ourselves, this time using 'float'.
	typedef boost::geometry::model::point<float, 3, boost::geometry::cs::cartesian> point_3d_t;
	typedef boost::geometry::model::polygon<point_3d_t> polygon_3d_t;

	polygon_3d_t poly3;
#if BOOST_VERSION <= 105200
	boost::geometry::ring_type<polygon_3d_t>::type &ring = boost::geometry::exterior_ring(poly3);
	boost::geometry::append(ring, boost::geometry::make<point_3d_t>(2, 2, 3));
	boost::geometry::append(ring, boost::geometry::make<point_3d_t>(2, 5, 6));
	boost::geometry::append(ring, boost::geometry::make<point_3d_t>(2, 8, 1));
	boost::geometry::append(ring, boost::geometry::make<point_3d_t>(2, 2, 3));
#else
	const double coor[][3] = { {2, 2, 3}, {2, 5, 6}, {2, 8, 1}, {2, 2, 3} };
	boost::geometry::assign_points(poly3, coor);
#endif

	std::cout << "3D: polygon: " << boost::geometry::dsv(poly3) << std::endl;

	std::cout << "3D: length: " << boost::geometry::length(poly3) << std::endl;  // Oops !!!: error.
	//std::cout << "3D: area: " << boost::geometry::area(poly3) << std::endl;

	//point_3d_t cent;
	//boost::geometry::centroid(poly3d, cent);
	//std::cout << "3D: centroid: " << boost::geometry::dsv(cent) << std::endl;

	std::cout << "point in polygon:" << std::boolalpha << boost::geometry::within(boost::geometry::make<point_3d_t>(2, 4, 3), poly3) << std::endl;
}

// REF [file] >> ${BOOST_HOME}/libs/geometry/example/06_a_transformation_example.cpp
void transform_2d()
{
	typedef boost::geometry::model::d2::point_xy<double> point_2d_t;

	point_2d_t p(1, 1);
	point_2d_t p2;

	// Example: translate a point over (5,5)
#if BOOST_VERSION <= 105200
	boost::geometry::strategy::transform::translate_transformer<point_2d_t, point_2d_t> translate(5, 5);
#else
	boost::geometry::strategy::transform::translate_transformer<double, 2, 2> translate(5, 5);
#endif

	boost::geometry::transform(p, p2, translate);
	std::cout << "transformed point " << boost::geometry::dsv(p2) << std::endl;

	// Transform a polygon.
#if BOOST_VERSION <= 105200
	point_2d_t poly, poly2;
#else
    boost::geometry::model::polygon<point_2d_t> poly, poly2;
#endif
	const double coor[][2] = { {0, 0}, {0, 7}, {2, 2}, {2, 0}, {0, 0} };
	// Note that for this syntax you have to include the two include files above (c_array.hpp).
	boost::geometry::assign_points(poly, coor);
	//read_wkt("POLYGON((0 0,0 7,4 2,2 0,0 0))", poly);
	boost::geometry::transform(poly, poly2, translate);

	std::cout << "source      polygon " << boost::geometry::dsv(poly) << std::endl;
	std::cout << "transformed polygon " << boost::geometry::dsv(poly2) << std::endl;

	// Many more transformations are possible:
	// - from Cartesian to Spherical coordinate systems and back.
	// - from Cartesian to Cartesian (mapping, affine transformations) and back (inverse).
	// - Map Projections.
	// - from Degree to Radian and back in spherical or geographic coordinate systems.
}

void overlay_polygon()
{
	typedef boost::geometry::model::d2::point_xy<double> point_2d_t;
	typedef boost::geometry::model::polygon<point_2d_t> polygon_2d_t;

#if defined(HAVE_SVG)
	std::ofstream stream("05_a_intersection_polygon_example.svg");
	boost::geometry::svg_mapper<point_2d_t> svg(stream, 500, 500);
#endif

	// Define a polygons and fill the outer rings.
	polygon_2d_t a;
	{
		const double c[][2] = {
			{160, 330}, {60, 260}, {20, 150}, {60, 40}, {190, 20}, {270, 130}, {260, 250}, {160, 330}
		};
		boost::geometry::assign_points(a, c);
	}
	boost::geometry::correct(a);
	std::cout << "A: " << boost::geometry::dsv(a) << std::endl;

	polygon_2d_t b;
	{
		const double c[][2] = {
			{300, 330}, {190, 270}, {150, 170}, {150, 110}, {250, 30}, {380, 50}, {380, 250}, {300, 330}
		};
		boost::geometry::assign_points(b, c);
	}
	boost::geometry::correct(b);
	std::cout << "B: " << boost::geometry::dsv(b) << std::endl;

#if defined(HAVE_SVG)
	svg.add(a);
	svg.add(b);

	svg.map(a, "opacity:0.6;fill:rgb(0,255,0);");
	svg.map(b, "opacity:0.6;fill:rgb(0,0,255);");
#endif

	// Calculate interesection(s).
	std::vector<polygon_2d_t> intersection;
	boost::geometry::intersection(a, b, intersection);

	std::cout << "Intersection of polygons A and B" << std::endl;
	BOOST_FOREACH(polygon_2d_t const &polygon, intersection)
	{
		std::cout << boost::geometry::dsv(polygon) << std::endl;

#if defined(HAVE_SVG)
		svg.map(polygon, "opacity:0.5;fill:none;stroke:rgb(255,0,0);stroke-width:6");
#endif
	}
}

// REF [file] >> ${BOOST_HOME}/libs/geometry/example/05_b_overlay_linestring_polygon_example.cpp
void overlay_polygon_linestring()
{
	typedef boost::geometry::model::d2::point_xy<double> point_2d_t;

	boost::geometry::model::linestring<point_2d_t> ls;
	{
		const double c[][2] = { {0, 1}, {2, 5}, {5, 3} };
		boost::geometry::assign_points(ls, c);
	}

	boost::geometry::model::polygon<point_2d_t> p;
	{
		const double c[][2] = { {3, 0}, {0, 3}, {4, 5}, {3, 0} };
		boost::geometry::assign_points(p, c);
	}
	boost::geometry::correct(p);

#if defined(HAVE_SVG)
	// Create SVG-mapper
	std::ofstream stream("05_b_overlay_linestring_polygon_example.svg");
	boost::geometry::svg_mapper<point_2d_t> svg(stream, 500, 500);
	// Determine extend by adding geometries
	svg.add(p);
	svg.add(ls);
	// Map geometries
	svg.map(ls, "opacity:0.6;stroke:rgb(255,0,0);stroke-width:2;");
	svg.map(p, "opacity:0.6;fill:rgb(0,0,255);");
#endif

	// Calculate intersection points (turn points).
#if BOOST_VERSION <= 105500
	typedef boost::geometry::detail::overlay::turn_info<point_2d_t> turn_info_t;
#else
	typedef boost::geometry::segment_ratio_type<point_2d_t, boost::geometry::detail::no_rescale_policy>::type segment_ratio_t;
	typedef boost::geometry::detail::overlay::turn_info<point_2d_t, segment_ratio_t> turn_info_t;
#endif

	std::vector<turn_info_t> turns;
	boost::geometry::detail::get_turns::no_interrupt_policy policy;
#if BOOST_VERSION <= 105500
	boost::geometry::get_turns<false, false, boost::geometry::detail::overlay::assign_null_policy>(ls, p, turns, policy);
#else
	boost::geometry::detail::no_rescale_policy rescale_policy;
	boost::geometry::get_turns<false, false, boost::geometry::detail::overlay::assign_null_policy>(ls, p, rescale_policy, turns, policy);
#endif

	std::cout << "Intersection of linestring/polygon" << std::endl;
	BOOST_FOREACH(turn_info_t const &turn, turns)
	{
		std::string action = "intersecting";
		if (turn.operations[0].operation == boost::geometry::detail::overlay::operation_intersection)
		{
			action = "entering";
		}
		else if (turn.operations[0].operation == boost::geometry::detail::overlay::operation_union)
		{
			action = "leaving";

		}
		std::cout << action << " polygon at " << boost::geometry::dsv(turn.point) << std::endl;

#if defined(HAVE_SVG)
		svg.map(turn.point, "fill:rgb(255,128,0);stroke:rgb(0,0,100);stroke-width:1");
		svg.text(turn.point, action, "fill:rgb(0,0,0);font-family:Arial;font-size:10px");
#endif
	}
}

// REF [file] >> ${BOOST_HOME}/libs/geometry/example/06_b_transformation_example.cpp
struct random_style
{
	random_style()
	: rng(static_cast<int>(std::time(0))), dist(0, 255), colour(rng, dist)
	{}

	std::string fill(double opacity = 1)
	{
		std::ostringstream oss;
		oss << "fill:rgba(" << colour() << "," << colour() << "," << colour() << "," << opacity << ");";
		return oss.str();
	}

	std::string stroke(int width, double opacity = 1)
	{
		std::ostringstream oss;
		oss << "stroke:rgba(" << colour() << "," << colour() << "," << colour() << "," << opacity << ");";
		oss << "stroke-width:" << width  << ";";
		return oss.str();
	}

	template <typename T>
	std::string text(T x, T y, std::string const &text)
	{
		std::ostringstream oss;
		oss << "<text x=\"" << static_cast<int>(x) - 90 << "\" y=\"" << static_cast<int>(y) << "\" font-family=\"Verdana\">" << text << "</text>";
		return oss.str();
	}

	boost::mt19937 rng;
	boost::uniform_int<> dist;
	boost::variate_generator<boost::mt19937 &, boost::uniform_int<> > colour;
};

// REF [file] >> ${BOOST_HOME}/libs/geometry/example/06_b_transformation_example.cpp
template <typename OutputStream>
struct svg_output
{
	svg_output(OutputStream &os, double opacity = 1) : os(os), opacity(opacity)
	{
		os << "<?xml version=\"1.0\" standalone=\"no\"?>\n"
			<< "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n"
			<< "\"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n"
			<< "<svg width=\"100%\" height=\"100%\" version=\"1.1\"\n"
			<< "xmlns=\"http://www.w3.org/2000/svg\">" << std::endl;
	}

	~svg_output()
	{
		os << "</svg>" << std::endl;
	}

	template <typename G>
	void put(G const &g, std::string const &label)
	{
		std::string style_str(style.fill(opacity) + style.stroke(5, opacity));
#if defined(HAVE_SVG)
		os << boost::geometry::svg(g, style_str) << std::endl;
#endif
		if (!label.empty())
		{
			typename boost::geometry::point_type<G>::type c;
			boost::geometry::centroid(g, c);
			os << style.text(static_cast<int>(boost::geometry::get<0>(c)), static_cast<int>(boost::geometry::get<1>(c)), label);
		}
	}

private:

	OutputStream &os;
	double opacity;
	random_style style;
};

// REF [file] >> ${BOOST_HOME}/libs/geometry/example/06_b_transformation_example.cpp
void affine_transform_2d()
{
	typedef boost::geometry::model::d2::point_xy<double> point_2d_t;

	const std::string file("06_b_transformation_example.svg");

	std::ofstream ofs(file.c_str());
	svg_output<std::ofstream> svg(ofs, 0.5);

	// G1 - create subject for affine transformations.
	boost::geometry::model::polygon<point_2d_t> g1;
	boost::geometry::read_wkt("POLYGON((50 250, 400 250, 150 50, 50 250))", g1);
	std::clog << "Source box:\t" << boost::geometry::dsv(g1) << std::endl;
	svg.put(g1, "g1");

	// G1 - Translate -> G2.
#if BOOST_VERSION <= 105200
	boost::geometry::strategy::transform::translate_transformer<point_2d_t, point_2d_t> translate(0, 250);
#else
    boost::geometry::strategy::transform::translate_transformer<double, 2, 2> translate(0, 250);
#endif
	boost::geometry::model::polygon<point_2d_t> g2;
	boost::geometry::transform(g1, g2, translate);
	std::clog << "Translated:\t" << boost::geometry::dsv(g2) << std::endl;
	svg.put(g2, "g2=g1.translate(0,250)");

	// G2 - Scale -> G3.
#if BOOST_VERSION <= 105200
	boost::geometry::strategy::transform::scale_transformer<point_2d_t, point_2d_t> scale(0.5, 0.5);
#else
	boost::geometry::strategy::transform::scale_transformer<double, 2, 2> scale(0.5, 0.5);
#endif
	boost::geometry::model::polygon<point_2d_t> g3;
	boost::geometry::transform(g2, g3, scale);
	std::clog << "Scaled:\t" << boost::geometry::dsv(g3) << std::endl;
	svg.put(g3, "g3=g2.scale(0.5,0.5)");

	// G3 - Combine rotate and translate -> G4.
#if BOOST_VERSION <= 105200
	boost::geometry::strategy::transform::rotate_transformer<point_2d_t, point_2d_t, boost::geometry::degree> rotate(45);
#else
	boost::geometry::strategy::transform::rotate_transformer<boost::geometry::degree, double, 2, 2> rotate(45);
#endif

	// Compose matrix for the two transformation.
	// Create transformer attached to the transformation matrix.
#if BOOST_VERSION <= 105200
	boost::geometry::strategy::transform::ublas_transformer<point_2d_t, point_2d_t, 2, 2> combined(boost::numeric::ublas::prod(rotate.matrix(), translate.matrix()));
	//boost::geometry::strategy::transform::ublas_transformer<point_2d_t, point_2d_t, 2, 2> combined(rotate.matrix());
#else
	boost::geometry::strategy::transform::ublas_transformer<double, 2, 2> combined(boost::numeric::ublas::prod(rotate.matrix(), translate.matrix()));
	//boost::geometry::strategy::transform::ublas_transformer<double, 2, 2> combined(rotate.matrix());
#endif

	// Apply transformation to subject geometry point-by-point.
	boost::geometry::model::polygon<point_2d_t> g4;
	boost::geometry::transform(g3, g4, combined);

	std::clog << "Rotated & translated:\t" << boost::geometry::dsv(g4) << std::endl;
	svg.put(g4, "g4 = g3.(rotate(45) * translate(0,250))");

	std::clog << "Saved SVG file:\t" << file << std::endl;
}

void graph_route()
{
	// ref: example
	throw std::runtime_error("Not yet implemented");
}

// [ref] http://www.boost.org/doc/libs/1_53_0/libs/geometry/doc/html/geometry/reference/algorithms/intersection.html
void intersection_2d()
{
	typedef boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double> > polygon;

    polygon green, blue;

    boost::geometry::read_wkt(
        "POLYGON((2 1.3, 2.4 1.7, 2.8 1.8, 3.4 1.2, 3.7 1.6, 3.4 2, 4.1 3, 5.3 2.6, 5.4 1.2, 4.9 0.8, 2.9 0.7, 2 1.3)"
            "(4.0 2.0, 4.2 1.4, 4.8 1.9, 4.4 2.2, 4.0 2.0))",
        green
    );

    boost::geometry::read_wkt(
        "POLYGON((4.0 -0.5, 3.5 1.0, 2.0 1.5, 3.5 2.0, 4.0 3.5, 4.5 2.0, 6.0 1.5, 4.5 1.0, 4.0 -0.5))",
        blue
    );

    std::deque<polygon> output;
    boost::geometry::intersection(green, blue, output);

    int i = 0;
    std::cout << "green && blue:" << std::endl;
    BOOST_FOREACH(polygon const &poly, output)
    {
        std::cout << i++ << ": " << boost::geometry::area(poly) << std::endl;
    }
}

void intersection_3d()
{
#if 0
	typedef boost::geometry::model::polygon<boost::geometry::model::d3::point_xyz<double> > polyhedron;

    polyhedron cube, line;

    boost::geometry::read_wkt(
        "POLYHEDRALSURFACE Z ("
            "((0 0 0, 0 1 0, 1 1 0, 1 0 0, 0 0 0)),"
            "((0 0 0, 0 1 0, 0 1 1, 0 0 1, 0 0 0)),"
            "((0 0 0, 1 0 0, 1 0 1, 0 0 1, 0 0 0)),"
            "((1 1 1, 1 0 1, 0 0 1, 0 1 1, 1 1 1)),"
            "((1 1 1, 1 0 1, 1 0 0, 1 1 0, 1 1 1)),"
            "((1 1 1, 1 1 0, 0 1 0, 0 1 1, 1 1 1))"
        ")",
        cube
    );

    boost::geometry::read_wkt(
        "LINESTRING(0.5 0.5 -1.0, 0.5 0.5 1.0)",
        line
    );

    std::deque<polyhedron> output;
    boost::geometry::intersection(cube, line, output);

    int i = 0;
    std::cout << "green && blue:" << std::endl;
    BOOST_FOREACH(polyhedron const &poly, output)
    {
#if 0
        std::cout << i++ << ": " << boost::geometry::area(poly) << std::endl;
#else
        //BOOST_FOREACH(point const &p, poly)
        //{
        //    std::cout << boost::geometry::dsv(p) << std::endl;
        //}
#endif
    }
#else
    throw std::runtime_error("Boost.Geometry is not yet working in 3D");
#endif
}

}  // namespace local
}  // unnamed namespace

void geometry()
{
    // examples.
    if (true)
    {
        local::point_2d();
        local::linestring_2d();
        local::linestring_3d();
        local::polygon_2d();
        local::polygon_3d();

        local::overlay_polygon();
        local::overlay_polygon_linestring();

        local::transform_2d();
        local::affine_transform_2d();

        local::graph_route();

        local::intersection_2d();
    }

    // extensions.
    {
        //local::intersection_3d();  // not working.
    }
}
