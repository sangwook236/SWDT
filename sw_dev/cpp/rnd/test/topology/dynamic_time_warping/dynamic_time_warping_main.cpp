//include "stdafx.h"
#include "../dynamic_time_warping_lib/DynamicTimeWarping.h"
#include <boost/timer/timer.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>


namespace {
namespace local {

double abs_distance(const double &val1, const double &val2)
{
	return std::fabs(val1 - val2);
}

void simple_scalar_dtw_example()
{
	const std::string filename1("./data/topology/trace0.csv");
	const std::string filename2("./data/topology/trace1.csv");

	std::vector<double> scalars1;
	std::vector<double> scalars2;
	{
#if defined(__GNUC__)
		std::ifstream strm(filename1.c_str());
#else
		std::ifstream strm(filename1);
#endif
		if (!strm.is_open())
		{
			std::cerr << "File not found: " << filename1 << std::endl;
			return;
		}

		double val;
		while (strm && !strm.eof())
		{
			strm >> val;
			scalars1.push_back(val);
		}

		if (!strm.eof())
		{
			std::cerr << "Fooey!" << std::endl;
		}
	}
	{
#if defined(__GNUC__)
		std::ifstream strm(filename2.c_str());
#else
		std::ifstream strm(filename2);
#endif
		if (!strm.is_open())
		{
			std::cerr << "File not found: " << filename2 << std::endl;
			return;
		}

		double val;
		while (strm && !strm.eof())
		{
			strm >> val;
			scalars2.push_back(val);
		}

		if (!strm.eof())
		{
			std::cerr << "Fooey!" << std::endl;
		}
	}

	// REF [algorithm] >> FastDTW algorithm in Java.
	// Warp Distance: 9.139400704860002.
	// Warp Path:     [(0,0),(0,1),(1,2),(2,3),...,(272,272),(273,272),(274,273),(274,274)].
	std::cout << "Scalar dynamic time warping (DTW) test ..." << std::endl;
	const std::size_t maximumWarpingDistance = 10;
	double dist;
	{
		boost::timer::auto_cpu_timer timer;
		dist = computeFastDynamicTimeWarping(scalars1, scalars2, maximumWarpingDistance, abs_distance);
	}
	std::cout << "Distance: " << dist << std::endl;
}

class Point
{
public:
	Point(double X, double Y, double Z)
	: x(X), y(Y), z(Z)
	{}

public:
	double x, y, z;
};

// Compute the L1 distance.
double L1_distance(const Point &p1, const Point &p2)
{
	return std::fabs(p1.x - p2.x) + std::fabs(p1.y - p2.y) + std::fabs(p1.z - p2.z);
}

// Compute Euclidean (L2) distance.
double L2_distance(const Point &p1, const Point &p2)
{
	return std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

void simple_vector_dtw_example()
{
	Point p1(1, 2, 3);
	Point p2(2, 3, 4);
	Point p3(3, 2, 3);
	Point p4(2, 2, 3);

	std::vector<Point> mainVec;
	std::vector<Point> testVec;
	mainVec.push_back(p1);
	mainVec.push_back(p2);
	testVec.push_back(p3);
	testVec.push_back(p4);

	std::cout << "Vector dynamic time warping (DTW) test ..." << std::endl;
	const std::size_t maximumWarpingDistance = 1;
	double dist;
	{
		boost::timer::auto_cpu_timer timer;
		//dist = computeFastDynamicTimeWarping(mainVec, testVec, maximumWarpingDistance, L1_distance);
		dist = computeFastDynamicTimeWarping(mainVec, testVec, maximumWarpingDistance, L2_distance);
	}
	std::cout << "Distance: " << dist << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_dynamic_time_warping {

}  // namespace my_dynamic_time_warping

int dynamic_time_warping_main(int argc, char *argv[])
{
	local::simple_scalar_dtw_example();
	//local::simple_vector_dtw_example();

	return 0;
}
