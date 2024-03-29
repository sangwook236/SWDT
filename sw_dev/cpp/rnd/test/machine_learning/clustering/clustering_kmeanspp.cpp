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

// REF [file] >> ${CPP_RND_HOME}/test/machine_learning/clustering/clustering_spectral_clustering.cpp
void kmeanspp_sample_1()
{
#if 1
	const std::string input_filename("./data/machine_learning/clustering/data.txt");
	const int num_points = 317;
	const int dim_features = 2;
	const int num_clusters = 3;
#elif 0
	const std::string input_filename("./data/machine_learning/clustering/circles.txt");
	const int num_points = 139;
	const int dim_features = 2;
	const int num_clusters = 3;
#elif 0
	const std::string input_filename("./data/machine_learning/clustering/processed_input.txt");
	const int num_points = 78;
	const int dim_features = 2;
	const int num_clusters = 3;
#endif
	const int num_attempts = 1000;

#if defined(__GNUC__)
	std::ifstream stream(input_filename.c_str(), std::ios::in);
#else
	std::ifstream stream(input_filename, std::ios::in);
#endif
	if (!stream.is_open())
	{
		std::cerr << "file not found: " << input_filename << std::endl;
		return;
	}

	std::vector<Scalar> points;
	points.reserve(num_points * dim_features);
	{
		int x, y;
		while (!stream.eof())
		{
			stream >> x >> y;
			if (!stream.good()) break;
			points.push_back(x);
			points.push_back(y);
		}
	}
	assert(num_points == points.size() / dim_features);

	//
	{
		std::vector<Scalar> cluster_centers(num_clusters * dim_features, 0);
		std::vector<int> assignments(num_points, -1);

		std::cout << "start clustering ..." << std::endl;
		Scalar cost;
		{
			boost::timer::auto_cpu_timer timer;

			// run k-means or k-means++.
			//cost = RunKMeans(num_points, num_clusters, dim_features, &points[0], num_attempts, &cluster_centers[0], &assignments[0]);
			cost = RunKMeansPlusPlus(num_points, num_clusters, dim_features, &points[0], num_attempts, &cluster_centers[0], &assignments[0]);
		}
		std::cout << "end clustering ..." << std::endl;

		// show results
		std::cout << "the final cost of the clustering = " << cost << std::endl;
		std::cout << "the locations of all cluster centers:" << std::endl;
		for (int k = 0; k < num_clusters; ++k)
		{
			for (int d = 0; d < dim_features; ++d)
				std::cout << cluster_centers[k * dim_features + d] << ", ";
			std::cout << std::endl;
		}
		std::cout << "the cluster that each point is assigned to:" << std::endl;
		for (int n = 0; n < num_points; ++n)
			std::cout << assignments[n] << ", ";
		std::cout << std::endl;
	}
}

void kmeanspp_sample_2()
{
	const int num_points = 4601;
	const int dim_features = 58;
	const int num_clusters = 10;
	const int num_attempts = 1000;

	const std::string input_filename("./data/machine_learning/clustering/spam_input.txt");
#if defined(__GNUC__)
	std::ifstream stream(input_filename.c_str());
#else
	std::ifstream stream(input_filename);
#endif
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

			// run k-means
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

			// run k-means++ on the given set of points. Set RunKMeans for info on the parameters.
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

}  // namespace local
}  // unnamed namespace

namespace my_clustering {

void kmeanspp()
{
	local::kmeanspp_sample_1();
	//local::kmeanspp_sample_2();
}

}  // namespace my_clustering
