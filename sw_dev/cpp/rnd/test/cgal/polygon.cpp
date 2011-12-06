#include "stdafx.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_2_algorithms.h>
#include <iostream>


namespace {

typedef CGAL::Exact_predicates_inexact_constructions_kernel kernel_type;
typedef kernel_type::Point_2 point_type;
typedef CGAL::Polygon_2<kernel_type> polygon_type;

void check_inside(point_type &pt, point_type *pgn_begin, point_type *pgn_end, kernel_type traits)
{
	std::cout << "The point " << pt;
	switch (CGAL::bounded_side_2(pgn_begin, pgn_end, pt, traits))
	{
	case CGAL::ON_BOUNDED_SIDE:
		std::cout << " is inside the polygon." << std::endl;
		break;
	case CGAL::ON_BOUNDARY:
		std::cout << " is on the polygon boundary." << std::endl;
		break;
	case CGAL::ON_UNBOUNDED_SIDE:
		std::cout << " is outside the polygon." << std::endl;
		break;
	}
}

void polygon2()
{
	point_type points[] = { point_type(0,0), point_type(5.1,0), point_type(1,1), point_type(0.5,6) };
	polygon_type pgn(points, points + 4);

	// check if the polygon is simple.
	std::cout << "The polygon is " << (pgn.is_simple() ? "" : "not ") << "simple." << std::endl;
	std::cout << "The polygon is " << (CGAL::is_simple_2(points, points + 4, kernel_type()) ? "" : "not ") << "simple." << std::endl;

	// check if the polygon is convex
	std::cout << "The polygon is " << (pgn.is_convex() ? "" : "not ") << "convex." << std::endl;

	// check if the polygon is simple.
	check_inside(point_type(0.5, 0.5), points, points + 4, kernel_type());
	check_inside(point_type(1.5, 2.5), points, points + 4, kernel_type());
	check_inside(point_type(2.5, 0), points, points + 4, kernel_type());
}

}  // unnamed namespace

void polygon()
{
	polygon2();
}
