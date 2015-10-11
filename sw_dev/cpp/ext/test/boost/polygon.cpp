#include <boost/polygon/polygon.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <list>
#include <set>
#include <cassert>


namespace {
namespace local {

// REF [file] >>
//  ${BOOST_HOME}/libs/polygon/example/gtl_custom_point.cpp
//  ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon.cpp
//  ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon_set.cpp
struct CPoint
{
    int x;
    int y;
};

// REF [file] >>
//  ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon.cpp
//  ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon_set.cpp
typedef std::list<CPoint> CPolygon;

// REF [file] >> ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon_set.cpp
typedef std::deque<CPolygon> CPolygonSet;

}  // namespace local
}  // unnamed namespace

namespace boost {
namespace polygon {

// REF [file] >>
//  ${BOOST_HOME}/libs/polygon/example/gtl_custom_point.cpp
//  ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon.cpp
//  ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon_set.cpp
template <>
struct geometry_concept<local::CPoint>
{
	typedef point_concept type;
};

// REF [file] >>
//  ${BOOST_HOME}/libs/polygon/example/gtl_custom_point.cpp
//  ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon.cpp
//  ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon_set.cpp
// Then we specialize the boost::polygon::point_traits for our point type.
template <>
struct point_traits<local::CPoint>
{
	typedef int coordinate_type;

	static inline coordinate_type get(const local::CPoint &point, boost::polygon::orientation_2d orient)
	{
		if (orient == boost::polygon::HORIZONTAL)
			return point.x;
		return point.y;
	}
};

// REF [file] >>
//  ${BOOST_HOME}/libs/polygon/example/gtl_custom_point.cpp
//  ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon.cpp
//  ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon_set.cpp
template <>
struct point_mutable_traits<local::CPoint>
{
    typedef int coordinate_type;

	static inline void set(local::CPoint &point, boost::polygon::orientation_2d orient, int value)
	{
		if (orient == boost::polygon::HORIZONTAL)
			point.x = value;
		else
			point.y = value;
	}

	static inline local::CPoint construct(const int x_value, const int y_value)
	{
		local::CPoint retval;
		retval.x = x_value;
		retval.y = y_value;
		return retval;
	}
};

// REF [file] >>
//  ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon.cpp
//  ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon_set.cpp
template <>
struct geometry_concept<local::CPolygon>
{
	typedef polygon_concept type;
};

// REF [file] >>
//	${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon.cpp
//  ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon_set.cpp
template <>
struct polygon_traits<local::CPolygon>
{
	typedef int coordinate_type;
	typedef local::CPolygon::const_iterator iterator_type;
	typedef local::CPoint point_type;

	// Get the begin iterator
	static inline iterator_type begin_points(const local::CPolygon &t)
	{
		return t.begin();
	}

	// Get the end iterator
	static inline iterator_type end_points(const local::CPolygon &t)
	{
		return t.end();
	}

	// Get the number of sides of the polygon
	static inline std::size_t size(const local::CPolygon &t)
	{
		return t.size();
	}

	// Get the winding direction of the polygon
	static inline winding_direction winding(const local::CPolygon &t)
	{
		return unknown_winding;
	}
};

// REF [file] >>
//  ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon.cpp
//  ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon_set.cpp
template <>
struct polygon_mutable_traits<local::CPolygon>
{
	// expects stl style iterators
	template <typename iT>
	static inline local::CPolygon & set_points(local::CPolygon &t, iT input_begin, iT input_end)
	{
		t.clear();
#if 0
        // REF [file] >> ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon.cpp
		t.insert(t.end(), input_begin, input_end);
#else
        // REF [file] >> ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon_set.cpp
		while (input_begin != input_end)
		{
			t.push_back(local::CPoint());
			boost::polygon::assign(t.back(), *input_begin);
			++input_begin;
		}
#endif
		return t;
	}

};

// REF [file] >> ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon_set.cpp
template <>
struct geometry_concept<local::CPolygonSet>
{
	typedef polygon_set_concept type;
};

// REF [file] >> ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon_set.cpp
//	next we map to the concept through traits.
template <>
struct polygon_set_traits<local::CPolygonSet>
{
	typedef int coordinate_type;
	typedef local::CPolygonSet::const_iterator iterator_type;
	typedef local::CPolygonSet operator_arg_type;

	static inline iterator_type begin(const local::CPolygonSet &polygon_set)
	{
		return polygon_set.begin();
	}
	static inline iterator_type end(const local::CPolygonSet &polygon_set)
	{
		return polygon_set.end();
	}

	// don't worry about these, just return false from them.
	static inline bool clean(const local::CPolygonSet &polygon_set)
	{
		return false;
	}
	static inline bool sorted(const local::CPolygonSet &polygon_set)
	{
		return false;
	}
};

// REF [file] >> ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon_set.cpp
template <>
struct polygon_set_mutable_traits<local::CPolygonSet>
{
	template <typename input_iterator_type>
	static inline void set(local::CPolygonSet &polygon_set, input_iterator_type input_begin, input_iterator_type input_end)
	{
		polygon_set.clear();
		// this is kind of cheesy. I am copying the unknown input geometry into my own polygon set and then calling get to populate the deque.
		boost::polygon::polygon_set_data<int> ps;
		ps.insert(input_begin, input_end);
		ps.get(polygon_set);
		// if you had your own odd-ball polygon set you would probably have to iterate through each polygon at this point and do something extra.
	}
};

}  // namespace polygon
}  // namespace boost

namespace {
namespace local {

using namespace boost::polygon::operators;

void point()
{
	const int x = 10;
	const int y = 20;
	boost::polygon::point_data<int> pt(x, y);
	assert(boost::polygon::x(pt) == 10);
	assert(boost::polygon::y(pt) == 20);

	// a quick primer in isotropic point access
	boost::polygon::orientation_2d orient = boost::polygon::HORIZONTAL;
	assert(boost::polygon::x(pt) == boost::polygon::get(pt, orient));

	orient = orient.get_perpendicular();
	assert(orient == boost::polygon::VERTICAL);
	assert(boost::polygon::y(pt) == boost::polygon::get(pt, orient));

	boost::polygon::set(pt, orient, 30);
	assert(boost::polygon::y(pt) == 30);

	// using some of the library functions
	boost::polygon::point_data<int> pt2(10, 30);
	assert(boost::polygon::equivalence(pt, pt2));

	boost::polygon::transformation<int> tr(boost::polygon::axis_transformation::SWAP_XY);
	boost::polygon::transform(pt, tr);
	assert(boost::polygon::equivalence(pt, boost::polygon::point_data<int>(30, 10)));

	boost::polygon::transformation<int> tr2 = tr.inverse();
	assert(tr == tr2);  // SWAP_XY is its own inverse transform

	boost::polygon::transform(pt, tr2);
	assert(boost::polygon::equivalence(pt, pt2));  // the two points are equal again

	boost::polygon::move(pt, orient, 10);  // move pt 10 units in y
	assert(boost::polygon::euclidean_distance(pt, pt2) == 10.0f);

	boost::polygon::move(pt, orient.get_perpendicular(), 10);  // move pt 10 units in x
	assert(boost::polygon::manhattan_distance(pt, pt2) == 20);
}

// REF [file] >> ${BOOST_HOME}/libs/polygon/example/gtl_custom_point.cpp
template <typename Point>
void custom_point()
{
	// constructing a boost::polygon point
	const int x = 10;
	const int y = 20;

	//Point pt(x, y);
	Point pt = boost::polygon::construct<Point>(x, y);
	assert(boost::polygon::x(pt) == 10);
	assert(boost::polygon::y(pt) == 20);

	// a quick primer in isotropic point access
	boost::polygon::orientation_2d orient = boost::polygon::HORIZONTAL;
	assert(boost::polygon::x(pt) == boost::polygon::get(pt, orient));

	orient = orient.get_perpendicular();
	assert(orient == boost::polygon::VERTICAL);
	assert(boost::polygon::y(pt) == boost::polygon::get(pt, orient));

	boost::polygon::set(pt, orient, 30);
	assert(boost::polygon::y(pt) == 30);

	// using some of the library functions
	//Point pt2(10, 30);
	Point pt2 = boost::polygon::construct<Point>(10, 30);
	assert(boost::polygon::equivalence(pt, pt2));

	boost::polygon::transformation<int> tr(boost::polygon::axis_transformation::SWAP_XY);
	boost::polygon::transform(pt, tr);
	assert(boost::polygon::equivalence(pt, boost::polygon::construct<Point>(30, 10)));

	boost::polygon::transformation<int> tr2 = tr.inverse();
	assert(tr == tr2);  // SWAP_XY is its own inverse transform

	boost::polygon::transform(pt, tr2);
	assert(boost::polygon::equivalence(pt, pt2));  // the two points are equal again

	boost::polygon::move(pt, orient, 10);  // move pt 10 units in y
	assert(boost::polygon::euclidean_distance(pt, pt2) == 10.0f);

	boost::polygon::move(pt, orient.get_perpendicular(), 10);  // move pt 10 units in x
	assert(boost::polygon::manhattan_distance(pt, pt2) == 20);
}

void polygon()
{
	typedef boost::polygon::polygon_data<int> Polygon;
	typedef boost::polygon::polygon_traits<Polygon>::point_type Point;

	// lets construct a 10x10 rectangle shaped polygon
	Point pts[] = {
		boost::polygon::construct<Point>(0, 0),
		boost::polygon::construct<Point>(10, 0),
		boost::polygon::construct<Point>(10, 10),
		boost::polygon::construct<Point>(0, 10)
	};
	Polygon poly;
	boost::polygon::set_points(poly, pts, pts + 4);

	// now lets see what we can do with this polygon
	assert(boost::polygon::area(poly) == 100.0f);
	assert(boost::polygon::contains(poly, boost::polygon::construct<Point>(5, 5)));
	assert(!boost::polygon::contains(poly, boost::polygon::construct<Point>(15, 5)));

	boost::polygon::rectangle_data<int> rect;
	assert(boost::polygon::extents(rect, poly));  // get bounding box of poly
	assert(boost::polygon::equivalence(rect, poly));  // hey, that's slick
	assert(boost::polygon::winding(poly) == boost::polygon::COUNTERCLOCKWISE);
	assert(boost::polygon::perimeter(poly) == 40.0f);

	// add 5 to all coords of poly
	boost::polygon::convolve(poly, boost::polygon::construct<Point>(5, 5));
	// multiply all coords of poly by 2
	boost::polygon::scale_up(poly, 2);

	boost::polygon::set_points(rect, boost::polygon::point_data<int>(10, 10), boost::polygon::point_data<int>(30, 30));
	assert(boost::polygon::equivalence(poly, rect));
}

// REF [file] >> ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon.cpp
template <typename Polygon>
void custom_polygon()
{
	typedef typename boost::polygon::polygon_traits<Polygon>::point_type Point;

	// lets construct a 10x10 rectangle shaped polygon
	Point pts[] = {
		boost::polygon::construct<Point>(0, 0),
		boost::polygon::construct<Point>(10, 0),
		boost::polygon::construct<Point>(10, 10),
		boost::polygon::construct<Point>(0, 10)
	};
	Polygon poly;
	boost::polygon::set_points(poly, pts, pts + 4);

	// now lets see what we can do with this polygon
	assert(boost::polygon::area(poly) == 100.0f);
	assert(boost::polygon::contains(poly, boost::polygon::construct<Point>(5, 5)));
	assert(!boost::polygon::contains(poly, boost::polygon::construct<Point>(15, 5)));

	boost::polygon::rectangle_data<int> rect;
	assert(boost::polygon::extents(rect, poly));  // get bounding box of poly
	assert(boost::polygon::equivalence(rect, poly));  // hey, that's slick
	assert(boost::polygon::winding(poly) == boost::polygon::COUNTERCLOCKWISE);
	assert(boost::polygon::perimeter(poly) == 40.0f);

	// add 5 to all coords of poly
	boost::polygon::convolve(poly, boost::polygon::construct<Point>(5, 5));
	// multiply all coords of poly by 2
	boost::polygon::scale_up(poly, 2);

	boost::polygon::set_points(rect, boost::polygon::point_data<int>(10, 10), boost::polygon::point_data<int>(30, 30));
	assert(boost::polygon::equivalence(poly, rect));
}

void polygon_set()
{
	typedef std::vector<boost::polygon::polygon_data<int> > PolygonSet;

	// lets declare ourselves a polygon set
	PolygonSet ps;
	ps += boost::polygon::rectangle_data<int>(0, 0, 10, 10);

	// now lets do something interesting
	PolygonSet ps2;
	ps2 += boost::polygon::rectangle_data<int>(5, 5, 15, 15);
	PolygonSet ps3;
	boost::polygon::assign(ps3, ps * ps2);  // woah, I just felt the room flex around me

	PolygonSet ps4;
	ps4 += ps + ps2;

	// assert that area of result is equal to sum of areas of input geometry minus the area of overlap between inputs
	assert(boost::polygon::area(ps4) == boost::polygon::area(ps) + boost::polygon::area(ps2) - boost::polygon::area(ps3));

	// I don't even see the code anymore, all I see is bounding box...interval...triangle

	// lets try that again in slow motion shall we?
	assert(boost::polygon::equivalence((ps + ps2) - (ps * ps2), ps ^ ps2));

	// hmm, subtracting the intersection from the union is equivalent to the xor, all this in one line of code,
	// now we're programming in bullet time (by the way, xor is implemented as one pass, not composition)

	// just for fun
	boost::polygon::rectangle_data<int> rect;
	assert(boost::polygon::extents(rect, ps ^ ps2));
	assert(boost::polygon::area(rect) == 225);
	assert(boost::polygon::area(rect ^ (ps ^ ps2)) == boost::polygon::area(rect) - boost::polygon::area(ps ^ ps2));
}

// REF [file] >> ${BOOST_HOME}/libs/polygon/example/gtl_custom_polygon_set.cpp
template <typename PolygonSet>
void custom_polygon_set()
{
	PolygonSet ps;
	ps += boost::polygon::rectangle_data<int>(0, 0, 10, 10);
	PolygonSet ps2;
	ps2 += boost::polygon::rectangle_data<int>(5, 5, 15, 15);
	PolygonSet ps3;
	boost::polygon::assign(ps3, ps * ps2);
	PolygonSet ps4;
	ps4 += ps + ps2;

	assert(boost::polygon::area(ps4) == boost::polygon::area(ps) + boost::polygon::area(ps2) - boost::polygon::area(ps3));
	assert(boost::polygon::equivalence((ps + ps2) - (ps * ps2), ps ^ ps2));

	boost::polygon::rectangle_data<int> rect;
	assert(boost::polygon::extents(rect, ps ^ ps2));
	assert(boost::polygon::area(rect) == 225);
	assert(boost::polygon::area(rect ^ (ps ^ ps2)) == boost::polygon::area(rect) - boost::polygon::area(ps ^ ps2));
}

// this function works with both the 90 and 45 versions of connectivity extraction algorithm
template <typename ce_type>
void connectivity_extraction()
{
	// first we create an object to do the connectivity extraction
	ce_type ce;

	// create some test data
	std::vector<boost::polygon::rectangle_data<int> > test_data;
	test_data.push_back(boost::polygon::rectangle_data<int>(10, 10, 90, 90));
	test_data.push_back(boost::polygon::rectangle_data<int>(0, 0, 20, 20));
	test_data.push_back(boost::polygon::rectangle_data<int>(80, 0, 100, 20));
	test_data.push_back(boost::polygon::rectangle_data<int>(0, 80, 20, 100));
	test_data.push_back(boost::polygon::rectangle_data<int>(80, 80, 100, 100));
	// There is one big square and four little squares covering each of its corners.

	for (unsigned int i = 0; i < test_data.size(); ++i)
	{
		// insert returns an id starting at zero and incrementing with each call
		assert(ce.insert(test_data[i]) == i);
	}
	// notice that ids returned by ce.insert happen to match index into vector of inputs in this case

	// make sure the vector graph has elements for our nodes
	std::vector<std::set<int> > graph(test_data.size());

	// populate the graph with edge data
	ce.extract(graph);

	// make a map type graph to compare results
	std::map<int, std::set<int> > map_graph;
	ce.extract(map_graph);

	assert(map_graph.size() && map_graph.size() == graph.size());
	for (unsigned int i = 0; i < graph.size(); ++i)
	{
		assert(graph[i] == map_graph[i]);
		if (i == 0)
			assert(graph[i].size() == 4);  // four little squares
		else
			assert(graph[i].size() == 1);  // each little toches the big square
	}
}

//just a little meta-programming to get things off on the right foot
template <typename T>
struct lookup_polygon_set_type
{
	typedef boost::polygon::polygon_set_data<int> type;
};

template <typename T, typename T2>
struct lookup_polygon_set_type<boost::polygon::property_merge_90<T, T2> >
{
	typedef boost::polygon::polygon_90_set_data<int> type;
};

// This function works with both the 90 and general versions of property merge/map overlay algorithm
template <typename pm_type>
void property_merge()
{
	std::vector<boost::polygon::rectangle_data<int> > test_data;
	test_data.push_back(boost::polygon::rectangle_data<int>(11, 10, 31, 30));
	test_data.push_back(boost::polygon::rectangle_data<int>(1, 0, 21, 20));
	test_data.push_back(boost::polygon::rectangle_data<int>(6, 15, 16, 25));

	pm_type pm;

	// insert our test geometry into the property merge algorithm
	for (unsigned int i = 0; i < test_data.size(); ++i)
	{
		pm.insert(test_data[i], i);  // notice I use the index as the property value
	}

	typedef typename lookup_polygon_set_type<pm_type>::type polygon_set_type;
	typedef std::map<std::set<int>, polygon_set_type> property_merge_result_type;

	std::set<int> key;

	// There are 8 different combinations of our input geometries null combination is not interesting, so really 7

	property_merge_result_type result;
	pm.merge(result);

	// lets enumerate boolean combinations of inputs (hold onto your hats)
	for (unsigned int i = 0; i < 8; ++i)
	{
		bool bits[3] = { bool(i & 1), bool(i & 2), bool(i & 4) };  // break out bit array
		polygon_set_type test_set;
		std::set<int> key;
		for (unsigned int j = 0; j < 3; ++j)
		{
			if (bits[j])
			{
				key.insert(key.end(), j);
				test_set += test_data[j];
			}
		}
		for (unsigned int j = 0; j < 3; ++j)
		{
			if (bits[j])
				test_set *= test_data[j];
		}
		for (unsigned int j = 0; j < 3; ++j)
		{
			if (!bits[j])
				test_set -= test_data[j];
		}

		if (test_set.empty())
		{
			// only the null combination should not exist
			assert(i == 0);
			// a combination that does not exist should not be present in result
			assert(result.find(key) == result.end());
		}
		else
		{
			assert(boost::polygon::equivalence(result[key], test_set));
		}
	}

	// Notice that we have to do O(2^n) booleans to compose the same result that is produced in one pass of property merge
	// given n input layers (8 = 2^3 in this example)
}

}  // namespace local
}  // unnamed namespace

void polygon()
{
    // examples
    {
        //std::cout << "Boost.Polygon ..." << std::endl;

        local::point();
        local::custom_point<local::CPoint>();
        local::polygon();
        //local::custom_polygon<local::CPolygon>();  // Oops !!! compile-time error : 'boost::polygon::size': ambiguous call to overloaded function.
        local::custom_polygon<boost::polygon::polygon_data<int> >();
        local::polygon_set();
        local::custom_polygon_set<local::CPolygonSet>();
        local::custom_polygon_set<boost::polygon::polygon_set_data<int> >();

        local::connectivity_extraction<boost::polygon::connectivity_extraction_90<int> >();
        local::connectivity_extraction<boost::polygon::connectivity_extraction_45<int> >();

        local::property_merge<boost::polygon::property_merge_90<int, int> >();
        local::property_merge<boost::polygon::property_merge<int, int> >();
	}
}
