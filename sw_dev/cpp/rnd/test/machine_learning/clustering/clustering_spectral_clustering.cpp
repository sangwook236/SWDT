//#include "stdafx.h"
#include "../kmeanspp_lib/KMeans.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>
#include <boost/timer/timer.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <list>
#include <vector>


namespace {
namespace local {

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_type;

struct square_Euclidean_distance_function
{
public:
	square_Euclidean_distance_function(const int dim_features)
	: dim_features_(dim_features)
	{}

public:
	double operator()(const std::vector<double> &points, const int i, const int j) const
	{
		const double *pt1 = &points[i * dim_features_], *pt2 = &points[j * dim_features_];
		double dist2 = 0.0;
		for (int d = 0; d < dim_features_; ++d, ++pt1, ++pt2)
			dist2 += (*pt1 - *pt2) * (*pt1 - *pt2);

		return dist2;
	}

private:
	const int dim_features_;
};

template<class DistanceFunction>
void construct_epsilon_neighborhood_graph(const double epsilon, const int num_points, const int dim_features, const std::vector<double> &points, DistanceFunction distance_func, matrix_type &weighted_adjacent_matrix)
{
	weighted_adjacent_matrix = matrix_type::Zero(num_points, num_points);

	const double epsilon2 = epsilon * epsilon;
	double dist2_ij;
	for (int i = 0; i < num_points; ++i)
		for (int j = 0; j < num_points; ++j)
		{
			if (i == j) continue;

			dist2_ij = distance_func(points, i, j);
			if (dist2_ij <= epsilon2)
				// TODO [choose] >>
				weighted_adjacent_matrix(i, j) = std::sqrt(dist2_ij);
				//weighted_adjacent_matrix(i, j) = dist2_ij;
		}
}

bool is_k_nearest_neighbor(const matrix_type& S, const int num_points, const int i, const int j, const int k)
{
    const double s_ij = S(i, j);
	int count = 0;
    for (int n = 0; n < num_points; ++n)
	{
		if (i == n) continue;

		if (S(i, n) < s_ij)
		{
			if (++count > k) return false;
		}
	}
    //return count <= k;
    return true;
}

template<class DistanceFunction>
void construct_k_neighbor_graph(const int k, const int num_points, const int dim_features, const std::vector<double> &points, DistanceFunction func, matrix_type &weighted_adjacent_matrix)
{
	matrix_type similarity_matrix = matrix_type::Zero(num_points, num_points);
	for (int i = 0; i < num_points; ++i)
		for (int j = 0; j < num_points; ++j)
		{
			if (i == j) continue;

			// TODO [choose] >>
			similarity_matrix(i, j) = distance_func(points, i, j);
			//similarity_matrix(i, j) = std::sqrt(distance_func(points, i, j));
		}

	weighted_adjacent_matrix = matrix_type::Zero(num_points, num_points);

	for (int i = 0; i < num_points; ++i)
		for (int j = 0; j < num_points; ++j)
		{
			if (i == j) continue;

			// TODO [choose] >>
            //if (is_k_nearest_neighbor(similarity_matrix, num_points, i, j, k))
            if (is_k_nearest_neighbor(similarity_matrix, num_points, i, j, k) || is_k_nearest_neighbor(similarity_matrix, num_points, j, i, k))
				weighted_adjacent_matrix(i, j) = similarity_matrix(i, j);
		}
}

template<class DistanceFunction>
void construct_fully_connected_graph(const double sigma, const int num_points, const int dim_features, const std::vector<double> &points, DistanceFunction distance_func, matrix_type &weighted_adjacent_matrix)
{
	weighted_adjacent_matrix = matrix_type::Zero(num_points, num_points);

	const double _2sigma2 = 2.0 * sigma * sigma;
	double dist_ij;
	for (int i = 0; i < num_points; ++i)
		for (int j = 0; j < num_points; ++j)
		{
			if (i == j) continue;

			// TODO [choose] >>
			dist_ij = std::exp(-distance_func(points, i, j) / _2sigma2);
			//const double dij = distance_func(points, i, j);
			//dist_ij = std::exp(-dij * dij / _2sigma2);

			weighted_adjacent_matrix(i, j) = dist_ij;
		}
}

void construct_diagonal_matrix(const matrix_type& weighted_adjacent_matrix, const int num_points, matrix_type &diagonal_matrix)
{
    double sum;
    diagonal_matrix = matrix_type::Zero(num_points, num_points);
    for (int i = 0; i < num_points; ++i)
	{
        sum = 0.0;
		for (int j = 0; j < num_points; ++j)
			sum += weighted_adjacent_matrix(i, j);

        diagonal_matrix(i, i) = sum;
    }
}

double run_spectral_clustering(const int method, const int num_points, const int dim_features, const int num_clusters, const int num_attempts, const std::vector<double> &points, std::vector<double> &cluster_centers, std::vector<int> &assignments)
{
	// construct weighted adjacent matrix, W.
	matrix_type weighted_adjacent_matrix;
#if 0
	const double epsilon = 10.0;
	construct_epsilon_neighborhood_graph(epsilon, num_points, dim_features, points, square_Euclidean_distance_function(dim_features), weighted_adjacent_matrix);
#elif 0
	const int k = 10;  // k-nearest neighbors
	construct_k_neighbor_graph(k, num_points, dim_features, points, square_Euclidean_distance_function(dim_features), weighted_adjacent_matrix);
#else
	const double sigma = 1.0;
	construct_fully_connected_graph(sigma, num_points, dim_features, points, square_Euclidean_distance_function(dim_features), weighted_adjacent_matrix);
#endif

	// construct diagonal matrix, D.
	matrix_type diagonal_matrix;
	construct_diagonal_matrix(weighted_adjacent_matrix, num_points, diagonal_matrix);

	// (unnormalized) graph Laplacian matrix, L = D - W.

	matrix_type D, U;
	std::vector<double> new_points;
	new_points.reserve(num_points * num_clusters);
	if (1 == method)  // unnormalized spectral clustering.
	{
		// eigen value decomposition, L * u = lambda * u.
		Eigen::SelfAdjointEigenSolver<matrix_type> evd(diagonal_matrix - weighted_adjacent_matrix);

		//D = evd.eigenvalues();
		U = evd.eigenvectors();
	}
	else if (2 == method)  // normalized spectral clustering according to Shi and Malik (2000).
	{
		// generalized eigen value decomposition, L * u = lambda * D * u.
		Eigen::GeneralizedSelfAdjointEigenSolver<matrix_type> evd(diagonal_matrix - weighted_adjacent_matrix, diagonal_matrix);

		//D = evd.eigenvalues();
		U = evd.eigenvectors();
	}
	else if (3 == method)  // normalized spectral clustering according to Ng et al. (2000).
	{
		// (normalized) graph Laplacian matrix:
		//	Lsym = D^-1/2 * L * D^-1/2

		// eigen value decomposition, Lsym * u = lambda * u
		matrix_type Dsqrt;
		diagonal_matrix.sqrt().evalTo(Dsqrt);
		const matrix_type invDsqrt = Dsqrt.inverse();
		Eigen::SelfAdjointEigenSolver<matrix_type> evd(invDsqrt * (diagonal_matrix - weighted_adjacent_matrix) * invDsqrt);

		//D = evd.eigenvalues();
		U = evd.eigenvectors();
	}
	else if (4 == method)  // normalized spectral clustering according to Ng et al. (2000).
	{
		// (normalized) graph Laplacian matrix:
		//	Lrw = D^-1 * L

		// eigen value decomposition, Lrw * u = lambda * u.
		Eigen::SelfAdjointEigenSolver<matrix_type> evd(diagonal_matrix.inverse() * (diagonal_matrix - weighted_adjacent_matrix));

		//D = evd.eigenvalues();
		U = evd.eigenvectors();
	}
	else
	{
		std::cerr << "improper method: " << method << std::endl;
		return 0.0;
	}

	// FIXME [check] >> The eigenvalues & the eigenvectors are sorted in increasing order. Is it correct?
	for (int n = 0; n < num_points; ++n)
		for (int d = 0; d < num_clusters; ++d)
			new_points.push_back(U(n, d));

	// run k-means or k-means++.
	//return RunKMeans(num_points, num_clusters, num_clusters, (Scalar *)&new_points[0], num_attempts, (Scalar *)&cluster_centers[0], &assignments[0]);
	return RunKMeansPlusPlus(num_points, num_clusters, num_clusters, (Scalar *)&new_points[0], num_attempts, (Scalar *)&cluster_centers[0], &assignments[0]);
}

// [ref]
//	"A tutorial on spectral clustering", U. von Luxburg, SC, 2007
//	"Normalized cuts and image segmentation", J. Shi and J. Malik, TPAMI, 2000
void spectral_clustering_sample_1()
{
	//const std::string input_filename("./machine_learning_data/clustering/circles.txt");
	const std::string input_filename("./machine_learning_data/clustering/data.txt");
	//const std::string input_filename("./machine_learning_data/clustering/processed_input.txt");

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

	std::list<double> point_list;
	{
		int x, y;
		while (!stream.eof())
		{
			stream >> x >> y;
			if (!stream.good()) break;
			point_list.push_back(x);
			point_list.push_back(y);
		}
	}

	std::vector<double> points(point_list.begin(), point_list.end());
	point_list.clear();

	const int dim_features = 2;
	const int num_clusters = 3;
	const int num_attempts = 1000;
	const int num_points = points.size() / dim_features;

	// [ref] ${CPP_RND_HOME}/test/machine_learning/clustering/clustering_kmeanspp.cpp
	{
		//std::vector<double> cluster_centers(num_clusters * dim_features, 0);
		std::vector<double> cluster_centers(num_clusters * num_clusters, 0);
		std::vector<int> assignments(num_points, -1);

		std::cout << "start spectral clustering ..." << std::endl;
		const int method = 2;  // 1 <= method <= 4. method 2 is recommended. the result of method 4 is not so good.
		Scalar cost;
		{
			boost::timer::auto_cpu_timer timer;
			cost = run_spectral_clustering(method, num_points, dim_features, num_clusters, num_attempts, points, cluster_centers, assignments);
		}
		std::cout << "end spectral clustering ..." << std::endl;

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

}  // namespace local
}  // unnamed namespace

namespace my_clustering {

void spectral_clustering()
{
	local::spectral_clustering_sample_1();
}

}  // namespace my_clustering
