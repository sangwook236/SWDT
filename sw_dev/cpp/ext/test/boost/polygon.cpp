#include <boost/polygon/polygon.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <list>
#include <set>
#include <cassert>


namespace {
namespace local {

struct CPoint
{
    int x;
    int y;
};

typedef std::list<CPoint> CPolygon;

typedef std::deque<CPolygon> CPolygonSet;

}  // local
}  // unnamed namespace


namespace boost { namespace polygon {

template <>
struct geometry_concept<local::CPoint>
{
	typedef point_concept type;
};

// Then we specialize the gtl point traits for our point type
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

template <>
struct point_mutable_traits<local::CPoint>
{
	static inline void set(local::CPoint &point, boost::polygon::orientation_2d orient, int value)
	{
		if (orient == boost::polygon::HORIZONTAL)
			point.x = value;
		else
			point.y = value;
	}

	static inline local::CPoint construct(int x_value, int y_value)
	{
		local::CPoint retval;
		retval.x = x_value;
		retval.y = y_value; 
		return retval;
	}
};

template <>
struct geometry_concept<local::CPolygon>
{
	typedef polygon_concept type;
};

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

template <>
struct polygon_mutable_traits<local::CPolygon>
{
	// expects stl style iterators
	template <typename iT>
	static inline local::CPolygon & set_points(local::CPolygon &t, iT input_begin, iT input_end)
	{
		t.clear();
		//t.insert(t.end(), input_begin, input_end);
		while (input_begin != input_end)
		{
			t.push_back(local::CPoint());
			boost::polygon::assign(t.back(), *input_begin);
			++input_begin;
		}
		return t;
	}

};

template <>
struct geometry_concept<local::CPolygonSet>
{
	typedef polygon_set_concept type;
};

//next we map to the concept through traits
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

	// don't worry about these, just return false from them
	static inline bool clean(const local::CPolygonSet &polygon_set)
	{
		return false;
	}
	static inline bool sorted(const local::CPolygonSet &polygon_set)
	{
		return false;
	}
};

template <>
struct polygon_set_mutable_traits<local::CPolygonSet>
{
	template <typename input_iterator_type>
	static inline void set(local::CPolygonSet &polygon_set, input_iterator_type input_begin, input_iterator_type input_end)
	{
		polygon_set.clear();
		// this is kind of cheesy. I am copying the unknown input geometry into my own polygon set and then calling get to populate the deque
		boost::polygon::polygon_set_data<int> ps;
		ps.insert(input_begin, input_end);
		ps.get(polygon_set);
		// if you had your own odd-ball polygon set you would probably have to iterate through each polygon at this point and do something extra
	}
};

} }

namespace {
namespace local {

namespace gtl = boost::polygon;
using namespace boost::polygon::operators;

void point()
{
	const int x = 10;
	const int y = 20;
	gtl::point_data<int> pt(x, y);
	assert(gtl::x(pt) == 10);
	assert(gtl::y(pt) == 20);

	// a quick primer in isotropic point access
	gtl::orientation_2d orient = gtl::HORIZONTAL;
	assert(gtl::x(pt) == gtl::get(pt, orient));

	orient = orient.get_perpendicular();
	assert(orient == gtl::VERTICAL);
	assert(gtl::y(pt) == gtl::get(pt, orient));

	gtl::set(pt, orient, 30);
	assert(gtl::y(pt) == 30);

	// using some of the library functions
	gtl::point_data<int> pt2(10, 30);
	assert(gtl::equivalence(pt, pt2));

	gtl::transformation<int> tr(gtl::axis_transformation::SWAP_XY);
	gtl::transform(pt, tr);
	assert(gtl::equivalence(pt, gtl::point_data<int>(30, 10)));

	gtl::transformation<int> tr2 = tr.inverse();
	assert(tr == tr2);  // SWAP_XY is its own inverse transform

	gtl::transform(pt, tr2);
	assert(gtl::equivalence(pt, pt2));  // the two points are equal again

	gtl::move(pt, orient, 10);  // move pt 10 units in y
	assert(gtl::euclidean_distance(pt, pt2) == 10.0f);

	gtl::move(pt, orient.get_perpendicular(), 10);  // move pt 10 units in x
	assert(gtl::manhattan_distance(pt, pt2) == 20);
}

template <typename Point>
void custom_point()
{
	// constructing a gtl point
	const int x = 10;
	const int y = 20;

	//Point pt(x, y);
	Point pt = gtl::construct<Point>(x, y);
	assert(gtl::x(pt) == 10);
	assert(gtl::y(pt) == 20);

	// a quick primer in isotropic point access
	gtl::orientation_2d orient = gtl::HORIZONTAL;
	assert(gtl::x(pt) == gtl::get(pt, orient));

	orient = orient.get_perpendicular();
	assert(orient == gtl::VERTICAL);
	assert(gtl::y(pt) == gtl::get(pt, orient));

	gtl::set(pt, orient, 30);
	assert(gtl::y(pt) == 30);

	// using some of the library functions
	//Point pt2(10, 30);
	Point pt2 = gtl::construct<Point>(10, 30);
	assert(gtl::equivalence(pt, pt2));

	gtl::transformation<int> tr(gtl::axis_transformation::SWAP_XY);
	gtl::transform(pt, tr);
	assert(gtl::equivalence(pt, gtl::construct<Point>(30, 10)));

	gtl::transformation<int> tr2 = tr.inverse();
	assert(tr == tr2);  // SWAP_XY is its own inverse transform

	gtl::transform(pt, tr2);
	assert(gtl::equivalence(pt, pt2));  // the two points are equal again

	gtl::move(pt, orient, 10);  // move pt 10 units in y
	assert(gtl::euclidean_distance(pt, pt2) == 10.0f);

	gtl::move(pt, orient.get_perpendicular(), 10);  // move pt 10 units in x
	assert(gtl::manhattan_distance(pt, pt2) == 20);
}

void polygon()
{
	typedef gtl::polygon_data<int> Polygon;
	typedef gtl::polygon_traits<Polygon>::point_type Point;

	// lets construct a 10x10 rectangle shaped polygon
	Point pts[] = {
		gtl::construct<Point>(0, 0),
		gtl::construct<Point>(10, 0),
		gtl::construct<Point>(10, 10),
		gtl::construct<Point>(0, 10)
	};
	Polygon poly;
	gtl::set_points(poly, pts, pts + 4);

	// now lets see what we can do with this polygon
	assert(gtl::area(poly) == 100.0f);
	assert(gtl::contains(poly, gtl::construct<Point>(5, 5)));
	assert(!gtl::contains(poly, gtl::construct<Point>(15, 5)));

	gtl::rectangle_data<int> rect;
	assert(gtl::extents(rect, poly));  // get bounding box of poly
	assert(gtl::equivalence(rect, poly));  // hey, that's slick
	assert(gtl::winding(poly) == gtl::COUNTERCLOCKWISE);
	assert(gtl::perimeter(poly) == 40.0f);

	// add 5 to all coords of poly
	gtl::convolve(poly, gtl::construct<Point>(5, 5));
	// multiply all coords of poly by 2
	gtl::scale_up(poly, 2);

	gtl::set_points(rect, gtl::point_data<int>(10, 10), gtl::point_data<int>(30, 30));
	assert(gtl::equivalence(poly, rect));
}

template <typename Polygon>
void custom_polygon()
{
	typedef typename gtl::polygon_traits<Polygon>::point_type Point;

	// lets construct a 10x10 rectangle shaped polygon
	Point pts[] = {
		gtl::construct<Point>(0, 0),
		gtl::construct<Point>(10, 0),
		gtl::construct<Point>(10, 10),
		gtl::construct<Point>(0, 10)
	};
	Polygon poly;
	gtl::set_points(poly, pts, pts + 4);

	// now lets see what we can do with this polygon
	assert(gtl::area(poly) == 100.0f);
	assert(gtl::contains(poly, gtl::construct<Point>(5, 5)));
	assert(!gtl::contains(poly, gtl::construct<Point>(15, 5)));

	gtl::rectangle_data<int> rect;
	assert(gtl::extents(rect, poly));  // get bounding box of poly
	assert(gtl::equivalence(rect, poly));  // hey, that's slick
	assert(gtl::winding(poly) == gtl::COUNTERCLOCKWISE);
	assert(gtl::perimeter(poly) == 40.0f);

	// add 5 to all coords of poly
	gtl::convolve(poly, gtl::construct<Point>(5, 5));
	// multiply all coords of poly by 2
	gtl::scale_up(poly, 2);

	gtl::set_points(rect, gtl::point_data<int>(10, 10), gtl::point_data<int>(30, 30));
	assert(gtl::equivalence(poly, rect));
}

void polygon_set()
{
	typedef std::vector<gtl::polygon_data<int> > PolygonSet;

	// lets declare ourselves a polygon set
	PolygonSet ps;
	ps += gtl::rectangle_data<int>(0, 0, 10, 10);

	// now lets do something interesting
	PolygonSet ps2;
	ps2 += gtl::rectangle_data<int>(5, 5, 15, 15);
	PolygonSet ps3;
	gtl::assign(ps3, ps * ps2);  // woah, I just felt the room flex around me

	PolygonSet ps4;
	ps4 += ps + ps2;

	// assert that area of result is equal to sum of areas of input geometry minus the area of overlap between inputs
	assert(gtl::area(ps4) == gtl::area(ps) + gtl::area(ps2) - gtl::area(ps3));

	// I don't even see the code anymore, all I see is bounding box...interval...triangle

	// lets try that again in slow motion shall we?
	assert(gtl::equivalence((ps + ps2) - (ps * ps2), ps ^ ps2));

	// hmm, subtracting the intersection from the union is equivalent to the xor, all this in one line of code,
	// now we're programming in bullet time (by the way, xor is implemented as one pass, not composition)  

	// just for fun
	gtl::rectangle_data<int> rect;
	assert(gtl::extents(rect, ps ^ ps2));
	assert(gtl::area(rect) == 225);
	assert(gtl::area(rect ^ (ps ^ ps2)) == gtl::area(rect) - gtl::area(ps ^ ps2)); 
}

template <typename PolygonSet>
void custom_polygon_set()
{
	PolygonSet ps;
	ps += gtl::rectangle_data<int>(0, 0, 10, 10);
	PolygonSet ps2;
	ps2 += gtl::rectangle_data<int>(5, 5, 15, 15);
	PolygonSet ps3;
	gtl::assign(ps3, ps * ps2); 
	PolygonSet ps4;
	ps4 += ps + ps2;

	assert(gtl::area(ps4) == gtl::area(ps) + gtl::area(ps2) - gtl::area(ps3));
	assert(gtl::equivalence((ps + ps2) - (ps * ps2), ps ^ ps2));

	gtl::rectangle_data<int> rect;
	assert(gtl::extents(rect, ps ^ ps2));
	assert(gtl::area(rect) == 225);
	assert(gtl::area(rect ^ (ps ^ ps2)) == gtl::area(rect) - gtl::area(ps ^ ps2)); 
}

// this function works with both the 90 and 45 versions of connectivity extraction algorithm
template <typename ce_type>
void connectivity_extraction()
{
	// first we create an object to do the connectivity extraction
	ce_type ce;

	// create some test data
	std::vector<gtl::rectangle_data<int> > test_data;
	test_data.push_back(gtl::rectangle_data<int>(10, 10, 90, 90));
	test_data.push_back(gtl::rectangle_data<int>(0, 0, 20, 20));
	test_data.push_back(gtl::rectangle_data<int>(80, 0, 100, 20));
	test_data.push_back(gtl::rectangle_data<int>(0, 80, 20, 100));
	test_data.push_back(gtl::rectangle_data<int>(80, 80, 100, 100));
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
	typedef gtl::polygon_set_data<int> type;
};

template <typename T, typename T2>
struct lookup_polygon_set_type<gtl::property_merge_90<T, T2> >
{ 
	typedef gtl::polygon_90_set_data<int> type;
};

// This function works with both the 90 and general versions of property merge/map overlay algorithm
template <typename pm_type>
void property_merge()
{
	std::vector<gtl::rectangle_data<int> > test_data;
	test_data.push_back(gtl::rectangle_data<int>(11, 10, 31, 30));
	test_data.push_back(gtl::rectangle_data<int>(1, 0, 21, 20));
	test_data.push_back(gtl::rectangle_data<int>(6, 15, 16, 25));

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
		bool bits[3] = { i & 1, i & 2, i & 4 };  // break out bit array
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
			assert(gtl::equivalence(result[key], test_set));
		}
	}

	// Notice that we have to do O(2^n) booleans to compose the same result that is produced in one pass of property merge
	// given n input layers (8 = 2^3 in this example)
}

}  // local
}  // unnamed namespace

void polygon()
{
	local::point();
	local::custom_point<local::CPoint>();
	local::polygon();
	local::custom_polygon<local::CPolygon>();
	local::custom_polygon<boost::polygon::polygon_data<int> >();
	local::polygon_set();
	local::custom_polygon_set<local::CPolygonSet>();
	local::custom_polygon_set<boost::polygon::polygon_set_data<int> >();

	local::connectivity_extraction<boost::polygon::connectivity_extraction_90<int> >();
	local::connectivity_extraction<boost::polygon::connectivity_extraction_45<int> >();

	local::property_merge<boost::polygon::property_merge_90<int, int> >();
	local::property_merge<boost::polygon::property_merge<int, int> >();
}
