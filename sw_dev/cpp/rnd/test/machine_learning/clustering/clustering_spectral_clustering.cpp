//#include "stdafx.h"
#include "../kmeanspp_lib/KMeans.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>
#include <boost/timer/timer.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


namespace {
namespace local {

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_type;

struct square_Euclidean_distance_function
{
public:
	square_Euclidean_distance_function(const std::vector<double> &points, const int dim_features)
	: points_(points), dim_features_(dim_features)
	{}

public:
	double operator()(const int i, const int j) const
	{
		const double *pt1 = &points_[i * dim_features_], *pt2 = &points_[j * dim_features_];
		double dist2 = 0.0;
		for (int d = 0; d < dim_features_; ++d, ++pt1, ++pt2)
			dist2 += (*pt1 - *pt2) * (*pt1 - *pt2);

		return dist2;
	}

private:
    const std::vector<double> &points_;
	const int dim_features_;
};

template<class DistanceFunction>
void construct_epsilon_neighborhood_graph(const double epsilon, const int num_points, const int dim_features, const std::vector<double> &points, DistanceFunction distance_func, matrix_type &W)
{
	W = matrix_type::Zero(num_points, num_points);

	const double epsilon2 = epsilon * epsilon;
	double dist2_ij;
	for (int i = 0; i < num_points; ++i)
		for (int j = 0; j < num_points; ++j)
		{
			if (i == j) continue;

			dist2_ij = distance_func(i, j);
			if (dist2_ij <= epsilon2)
				// TODO [choose] >>
				W(i, j) = std::sqrt(dist2_ij);
				//W(i, j) = dist2_ij;
		}
}

bool is_k_nearest_neighbor(const matrix_type &S, const int num_points, const int i, const int j, const int k)
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
void construct_k_neighbor_graph(const int k, const int num_points, const int dim_features, const std::vector<double> &points, DistanceFunction distance_func, matrix_type &W)
{
	matrix_type similarity_matrix = matrix_type::Zero(num_points, num_points);
	for (int i = 0; i < num_points; ++i)
		for (int j = 0; j < num_points; ++j)
		{
			if (i == j) continue;

			// TODO [choose] >>
			similarity_matrix(i, j) = distance_func(i, j);
			//similarity_matrix(i, j) = std::sqrt(distance_func(i, j));
		}

	W = matrix_type::Zero(num_points, num_points);

	for (int i = 0; i < num_points; ++i)
		for (int j = 0; j < num_points; ++j)
		{
			if (i == j) continue;

			// TODO [choose] >>
            //if (is_k_nearest_neighbor(similarity_matrix, num_points, i, j, k))
            if (is_k_nearest_neighbor(similarity_matrix, num_points, i, j, k) || is_k_nearest_neighbor(similarity_matrix, num_points, j, i, k))
				W(i, j) = similarity_matrix(i, j);
		}
}

template<class DistanceFunction>
void construct_fully_connected_graph(const double sigma, const int num_points, const int dim_features, const std::vector<double> &points, DistanceFunction distance_func, matrix_type &W)
{
	W = matrix_type::Zero(num_points, num_points);

	const double _2sigma2 = 2.0 * sigma * sigma;
	double dist_ij;
	for (int i = 0; i < num_points; ++i)
		for (int j = 0; j < num_points; ++j)
		{
			if (i == j) continue;

			// TODO [choose] >>
			dist_ij = std::exp(-distance_func(i, j) / _2sigma2);
			//const double dij = distance_func(i, j);
			//dist_ij = std::exp(-dij * dij / _2sigma2);

			W(i, j) = dist_ij;
		}
}

void construct_diagonal_matrix(const matrix_type& W, const int num_points, matrix_type &D)
{
    D = matrix_type::Zero(num_points, num_points);

    double sum;
	for (int i = 0; i < num_points; ++i)
	{
        sum = 0.0;
		for (int j = 0; j < num_points; ++j)
			sum += W(i, j);

        D(i, i) = sum;
    }
}

double run_spectral_clustering(const int clustering_method, const matrix_type &W, const int num_points, const int dim_features, const int num_clusters, const int num_attempts, std::vector<double> &cluster_centers, std::vector<int> &assignments)
{
	// Construct diagonal matrix, D.
	matrix_type D;
	construct_diagonal_matrix(W, num_points, D);

	// (Nnnormalized) graph Laplacian matrix, L = D - W.

	matrix_type eigVals, eigVecs;
	std::vector<double> new_points;
	new_points.reserve(num_points * num_clusters);
	if (1 == clustering_method)  // Nnnormalized spectral clustering.
	{
		// Eigen value decomposition, L * u = lambda * u.
		Eigen::SelfAdjointEigenSolver<matrix_type> evd(D - W);

		//eigVals = evd.eigenvalues();
		eigVecs = evd.eigenvectors();
	}
	else if (2 == clustering_method)  // Normalized spectral clustering according to Shi and Malik (2000).
	{
		// Generalized eigen value decomposition, L * u = lambda * D * u.
		Eigen::GeneralizedSelfAdjointEigenSolver<matrix_type> evd(D - W, D);

		//eigVals = evd.eigenvalues();
		eigVecs = evd.eigenvectors();
	}
	else if (3 == clustering_method)  // Normalized spectral clustering according to Ng et al. (2000).
	{
		// (Normalized) graph Laplacian matrix, Lsym = D^-1/2 * L * D^-1/2.

		// Eigen value decomposition, Lsym * u = lambda * u
		matrix_type Dsqrt;
		D.sqrt().evalTo(Dsqrt);
		//Eigen::MatrixSquareRoot<matrix_type>(D).compute(Dsqrt);
		const matrix_type invDsqrt = Dsqrt.inverse();
		Eigen::SelfAdjointEigenSolver<matrix_type> evd(invDsqrt * (D - W) * invDsqrt);

		//eigVals = evd.eigenvalues();
		eigVecs = evd.eigenvectors().leftCols(num_clusters);

		// TODO [check] >> Normalization of each row is required?
		const matrix_type normvec = eigVecs.rowwise().norm();
		for (int n = 0; n < num_points; ++n)
			eigVecs.row(n) /= normvec(n);
	}
	else if (4 == clustering_method)  // Normalized spectral clustering according to Ng et al. (2000).
	{
		// (Normalized) graph Laplacian matrix, Lrw = D^-1 * L

		// Eigen value decomposition, Lrw * u = lambda * u.
		Eigen::SelfAdjointEigenSolver<matrix_type> evd(D.inverse() * (D - W));

		//eigVals = evd.eigenvalues();
		eigVecs = evd.eigenvectors().leftCols(num_clusters);

		const matrix_type normvec = eigVecs.rowwise().norm();
		for (int n = 0; n < num_points; ++n)
			eigVecs.row(n) /= normvec(n);
	}
	else
	{
		std::cerr << "improper clustering method: " << clustering_method << std::endl;
		return 0.0;
	}

	// FIXME [check] >> The eigenvalues & the eigenvectors are sorted in increasing order. Is it correct?
	for (int n = 0; n < num_points; ++n)
		for (int d = 0; d < num_clusters; ++d)
			new_points.push_back(eigVecs(n, d));

	// Run k-means or k-means++.
	//return RunKMeans(num_points, num_clusters, num_clusters, (Scalar *)&new_points[0], num_attempts, (Scalar *)&cluster_centers[0], &assignments[0]);
	return RunKMeansPlusPlus(num_points, num_clusters, num_clusters, (Scalar *)&new_points[0], num_attempts, (Scalar *)&cluster_centers[0], &assignments[0]);
}

// REF [paper] >>
//	"A tutorial on spectral clustering", U. von Luxburg, SC, 2007
//	"Normalized cuts and image segmentation", J. Shi and J. Malik, TPAMI, 2000
void spectral_clustering_sample_1()
{
#if 1
	const std::string input_filename("./data/machine_learning/clustering/data.txt");
	const int num_points = 317;
	const int dim_features = 2;
	const int num_clusters = 3;

	// TODO [adjust] >> Epsilon & k have to be adjusted. results of spectral clustering is sensitive to epsilon, k, & sigma.
	const double epsilon = 7.0;  // For graph construction method 1.
	const int k = 10;  // k-nearest neighbors. for graph construction method 2.
	const double sigma = 1.0;  // For graph construction method 3.

	// clustering_method 1 ~ 3: good when sigma = 1.0.
	// clustering_method 4: not so good.
#elif 0
	const std::string input_filename("./data/machine_learning/clustering/circles.txt");
	const int num_points = 139;
	const int dim_features = 2;
	const int num_clusters = 3;

	// TODO [adjust] >> Epsilon & k have to be adjusted. results of spectral clustering is sensitive to epsilon, k, & sigma.
	const double epsilon = 4.0;  // For graph construction method 1.
	const int k = 10;  // k-nearest neighbors. for graph construction method 2.
	const double sigma = 2.0;  // For graph construction method 3.

	// Clustering_method 1 ~ 3: good when sigma = 2.0.
	// Clustering_method 4: good when sigma = 1.0.
#elif 0
	const std::string input_filename("./data/machine_learning/clustering/processed_input.txt");
	const int num_points = 78;
	const int dim_features = 2;
	const int num_clusters = 3;

	// TODO [adjust] >> Epsilon & k have to be adjusted. results of spectral clustering is sensitive to epsilon, k, & sigma.
	const double epsilon = 100.0;  // For graph construction method 1.
	const int k = 10;  // k-nearest neighbors. for graph construction method 2.
	const double sigma = 100.0;  // For graph construction method 3.

	// Clustering_method 1 ~ 3: good when sigma = 100.0.
	// Clustering_method 4: good when sigma = 70.0.
#endif
	const int num_attempts = 1000;
	const int clustering_method = 2;  // 1 <= clustering_method <= 4. clustering_method 2 is recommended.
	const int graph_construction_method = 3;  // 1 <= graph_construction_method <= 3.

	std::vector<double> points;
	points.reserve(num_points * dim_features);
	{
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
		assert(points.size() == num_points * dim_features);
	}

	// Construct weighted adjacent matrix, W.
	matrix_type weighted_adjacent_matrix;
	if (1 == graph_construction_method)
		construct_epsilon_neighborhood_graph(epsilon, num_points, dim_features, points, square_Euclidean_distance_function(points, dim_features), weighted_adjacent_matrix);
	else if (2 == graph_construction_method)
		construct_k_neighbor_graph(k, num_points, dim_features, points, square_Euclidean_distance_function(points, dim_features), weighted_adjacent_matrix);
	else if (3 == graph_construction_method)
		construct_fully_connected_graph(sigma, num_points, dim_features, points, square_Euclidean_distance_function(points, dim_features), weighted_adjacent_matrix);
	else
	{
		std::cerr << "improper graph construction method: " << graph_construction_method << std::endl;
		return;
	}

	// REF [file] >> ${CPP_RND_HOME}/test/machine_learning/clustering/clustering_kmeanspp.cpp
	{
		//std::vector<double> cluster_centers(num_clusters * dim_features, 0);
		std::vector<double> cluster_centers(num_clusters * num_clusters, 0);
		std::vector<int> assignments(num_points, -1);

		std::cout << "start spectral clustering ..." << std::endl;
		Scalar cost;
		{
			boost::timer::auto_cpu_timer timer;
			cost = run_spectral_clustering(clustering_method, weighted_adjacent_matrix, num_points, dim_features, num_clusters, num_attempts, cluster_centers, assignments);
		}
		std::cout << "end spectral clustering ..." << std::endl;

		// Show results.
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
