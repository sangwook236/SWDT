//#include "stdafx.h"
#include "../kmeanspp_lib/KMeans.h"
#include <boost/timer/timer.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cassert>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_clustering {

void kmeanspp()
{
	const int num_points = 4601;
	const int dim_features = 58;
	const int num_clusters = 10;
	const int num_attempts = 1000;

	const std::string input_filename("./machine_learning_data/clustering/spam_input.txt");
	std::ifstream stream(input_filename);
	if (!stream.is_open())
	{
		std::cerr << "file not found: " << input_filename << std::endl;
		return;
	}

	std::vector<Scalar> points;
	points.reserve(num_points * dim_features);
	{
		Scalar val;
		while (!stream.eof())
		{
			stream >> val;
			if (!stream.good()) break;
			points.push_back(val);
		}
	}
	assert(points.size() == num_points * dim_features);

	{
		std::vector<Scalar> centers(num_clusters * dim_features, 0);
		std::vector<int> assignments(num_points, -1);

		Scalar cost;
		{
			boost::timer::auto_cpu_timer timer;
			cost = RunKMeans(num_points, num_clusters, dim_features, &points[0], num_attempts, &centers[0], &assignments[0]);
		}

		// show results
		std::cout << "the final cost of the clustering = " << cost << std::endl;
		std::cout << "the locations of all cluster centers:" << std::endl;
		for (int k = 0; k < num_clusters; ++k)
		{
			for (int d = 0; d < dim_features; ++d)
				std::cout << centers[k * dim_features + d] << ", ";
			std::cout << std::endl;
		}
		std::cout << "the cluster that each point is assigned to:" << std::endl;
		//for (int n = 0; n < num_points; ++n)
		//	std::cout << assignments[n] << ", ";
		//std::cout << std::endl;
	}

	{
		std::vector<Scalar> centers(num_clusters * dim_features, 0);
		std::vector<int> assignments(num_points, -1);

		Scalar cost;
		{
			boost::timer::auto_cpu_timer timer;
	
			// Runs k-means++ on the given set of points. Set RunKMeans for info on the parameters.
			cost = RunKMeansPlusPlus(num_points, num_clusters, dim_features, &points[0], num_attempts, &centers[0], &assignments[0]);
		}

		// show results
		std::cout << "the final cost of the clustering = " << cost << std::endl;
		std::cout << "the locations of all cluster centers:" << std::endl;
		for (int k = 0; k < num_clusters; ++k)
		{
			for (int d = 0; d < dim_features; ++d)
				std::cout << centers[k * dim_features + d] << ", ";
			std::cout << std::endl;
		}
		std::cout << "the cluster that each point is assigned to:" << std::endl;
		//for (int n = 0; n < num_points; ++n)
		//	std::cout << assignments[n] << ", ";
		//std::cout << std::endl;
	}
}

}  // namespace my_clustering
