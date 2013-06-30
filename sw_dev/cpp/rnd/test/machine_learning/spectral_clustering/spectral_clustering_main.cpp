//#include "stdafx.h"
#include <spectral.h>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/foreach.hpp>
#include <boost/timer/timer.hpp>
#include <string>
#include <fstream>
#include <iostream>


namespace {
namespace local {

// [ref] ${SPECTRAL_CLUSTERING_HOME}/src/SpectralClustering.cpp
void simple_example(int argc, char *argv[])
{
#if 0
	unsigned int x, y, num_clusters, num_neighbors, num_points, retries;
	double threshold, sigma2;
	std::string method;
	// Declare the supported options.
	boost::program_options::options_description desc("Program options");
	desc.add_options()
		("help,h", "describe program usage")
		("clusters,k", boost::program_options::value<unsigned int>(&num_clusters)->default_value(1), "number of clusters to attempt")
		("neighbors,e", boost::program_options::value<unsigned int>(&num_neighbors)->default_value(1), "number of neighbors to consider in the graph")
		("sigma2,s", boost::program_options::value<double>(&sigma2)->default_value(10000), "fallout speed of gaussian metric")
		("threshold,t", boost::program_options::value<double>(&threshold)->default_value(0.01), "k-means clustering threshold")
		("method", boost::program_options::value<std::string>(&method)->default_value("spectral"), "method, either spectral or kmeans")
		("retries", boost::program_options::value<unsigned int>(&retries)->default_value(50000), "number of k-means attempts at making k clusters before giving up")
		;

	boost::program_options::variables_map vm;
	boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
	boost::program_options::notify(vm);

	if (vm.count("help") || (method != "spectral" && method != "kmeans"))
	{
		std::cout << desc << std::endl;
		return;
	}

	vpoint points;
	while (std::cin >> x >> y) points.push_back(point(x, y));
	num_points = points.size();
#else
	const unsigned int num_clusters = 3;  // number of clusters to attempt
	const unsigned int num_neighbors = 4;  // number of neighbors to consider in the graph
	const double sigma2 = 10000;  // fallout speed of Gaussian metric
	const double threshold = 0.01;  // k-means clustering threshold
	const unsigned int retries = 50000;  // number of k-means attempts at making k clusters before giving up
	const std::string method = "kmeans";  // method, either spectral or kmeans

	//const std::string input_filename("./machine_learning_data/spectral_clustering/circles.txt");
	const std::string input_filename("./machine_learning_data/spectral_clustering/data.txt");
	//const std::string input_filename("./machine_learning_data/spectral_clustering/processed_input.txt");
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

	vpoint points;
	{
		unsigned int x, y;
		while (!stream.eof())
		{
			stream >> x >> y;
			if (!stream.good()) break;
			points.push_back(point(x, y));
		}
	}
	const unsigned int num_points = points.size();
#endif

	// run clustering
	vcluster clusters;
	std::cout << "start clustering ..." << std::endl;
	{
		boost::timer::auto_cpu_timer timer;

		clusters = (method == "spectral") ?
			spectral(points, num_clusters, threshold, num_neighbors, sigma2, retries) :
			just_k_means(points, num_clusters, threshold, retries);
	}
	std::cout << "end clustering ..." << std::endl;

	// display result
	std::cout << "number of points = " << num_points << std::endl;
	std::cout << "number of clusters to attempt = " << num_clusters << std::endl;
	std::cout << "clusters = " << std::endl;
	BOOST_FOREACH(int x, clusters) { std::cout << x << " "; }
	std::cout << std::endl;
	//std::cout << "points = " << std::endl;
	//BOOST_FOREACH(int x, points) { std::cout << x.first << " " << x.second << std::endl; }
}

}  // namespace local
}  // unnamed namespace

namespace my_spectral_clustering {

}  // namespace my_spectral_clustering

int spectral_clustering_main(int argc, char *argv[])
{
	local::simple_example(argc, argv);

	return 0;
}
