//#include "stdafx.h"
#include "../kdtree_lib/kdtree.h"
#include <boost/timer/timer.hpp>
#include <iostream>
#include <ctime>
#include <cassert>


namespace {
namespace local {

// ${KDTREE_HOME}/examples/test.c.
bool example1()
{
	const int vcount = 1000;

	std::cout << "inserting " << vcount << " random vectors... " << std::endl;

	// create a k-d tree for 3-dimensional points.
	kdtree *kd = kd_create(3);

	{
		boost::timer::auto_cpu_timer timer;

		for (int i = 0; i < vcount; ++i)
		{
			const double x = ((double)std::rand() / RAND_MAX) * 200.0 - 100.0;
			const double y = ((double)std::rand() / RAND_MAX) * 200.0 - 100.0;
			const double z = ((double)std::rand() / RAND_MAX) * 200.0 - 100.0;

			const int retval = kd_insert3(kd, x, y, z, 0);
			assert(0 == retval);
		}
	}

	{
		boost::timer::cpu_timer timer;

		kdres *set = kd_nearest_range3(kd, 0.0, 0.0, 0.0, 40.0);
		
		std::cout << "range query returned " << kd_res_size(set) << " items" << std::endl;
		//const boost::timer::cpu_times elapsed_times(timer.elapsed());
		//std::cout << "elpased time : " << (elapsed_times.system + elapsed_times.user) << " sec" << std::endl;
		std::cout << timer.format() << std::endl;

		kd_res_free(set);
	}

	kd_free(kd);

	return true;
}

// returns the distance squared between two dims-dimensional double arrays.
double dist_sq(double *a1, double *a2, int dims)
{
	double dist_sq = 0, diff;
	while (--dims >= 0)
	{
		diff = (a1[dims] - a2[dims]);
		dist_sq += diff * diff;
	}
	return dist_sq;
}

// get a random double between -10 and 10.
double rd()
{
	return ((double)std::rand() / RAND_MAX) * 20.0 - 10.0;
}

// ${KDTREE_HOME}/examples/test2.c.
bool example2()
{
	const int num_pts = 1000;

	char *data = new char [num_pts];
	if (NULL == data)
	{
		std::cerr << "memory allocation failed" << std::endl;
		return false;
	}

	std::srand((unsigned int)std::time(NULL));

	// create a k-d tree for 3-dimensional points.
	kdtree *ptree = kd_create(3);

	// add some random nodes to the tree (assert nodes are successfully inserted).
	for (int i = 0; i < num_pts; ++i)
	{
		data[i] = 'a' + i;
		const int retval = kd_insert3(ptree, rd(), rd(), rd(), &data[i]);
		assert(0 == retval);
	}

	// find points closest to the origin and within distance radius.
	double pt[3] = { 0.0, 0.0, 5.0 };
	const double radius = 5.0;
	kdres *presults = kd_nearest_range(ptree, pt, radius);

	// print out all the points found in results.
	std::cout << "found " << kd_res_size(presults) << " results: " << std::endl;

	double pos[3], dist;
	while (!kd_res_end(presults))
	{
		// get the data and position of the current result item.
		const char *pch = (char *)kd_res_item(presults, pos);

		// compute the distance of the current result from the pt.
		dist = std::sqrt(dist_sq(pt, pos, 3));

		// print out the retrieved data.
		std::cout << "node at (" << pos[0] << ", " << pos[1] << ", " << pos[2] << ") is " << dist << " away and has data = " << *pch << std::endl; 

		// go to the next entry.
		kd_res_next(presults);
	}

	// free our tree, results set, and other allocated memory.
	delete [] data;
	kd_res_free(presults);
	kd_free(ptree);

	return true;
}

}  // namespace local
}  // unnamed namespace

namespace my_kdtree {

}  // namespace my_kdtree

int kdtree_main(int argc, char *argv[])
{
	local::example1();
	local::example2();

	return 0;
}
