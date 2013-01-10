#include <vl/kmeans.h>
#include <vl/ikmeans.h>
#include <vl/hikmeans.h>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_vlfeat {

void keams()
{
	const vl_size data_dim = 2;
	const vl_size num_data = 1000;
	const vl_size num_clusters = 3;
	const vl_size num_max_iterations = 100;

	float data[data_dim * num_data] = { 0.0f, };
	for (vl_size i = 0; i < data_dim * num_data; ++i)
		data[i] = (float)std::rand() / RAND_MAX;

	//
	std::cout << "start processing ..." << std::endl;

	const VlKMeansAlgorithm algorithm = VlKMeansLloyd;
	const VlVectorComparisonType distance = VlDistanceL2;

	VlKMeans *kmeans = vl_kmeans_new(algorithm, distance);

	//
	vl_kmeans_seed_centers_with_rand_data(kmeans, (void *)data, data_dim, num_data, num_clusters);
	//vl_kmeans_seed_centers_plus_plus(kmeans, (void *)data, data_dim, num_data, num_clusters);

	vl_kmeans_set_max_num_iterations(kmeans, num_max_iterations);
	const double energy = vl_kmeans_refine_centers(kmeans, data, num_data);

	//
	const vl_size num_iterations = vl_kmeans_get_num_repetitions(kmeans);
	const vl_type data_type = vl_kmeans_get_data_type(kmeans);
	const float *centers = (float *)vl_kmeans_get_centers(kmeans);
	for (int i = 0; i < num_data; ++i)
	{
		std::cout << '(';
		for (int j = 0; j < data_dim; ++j)
		{
			if (j) std::cout << ',';
			std::cout << data[i * data_dim + j];
		}
		std::cout << ')' << std::endl;
	}

	std::cout << "end processing ..." << std::endl;

	//
	if (kmeans)
	{
		vl_kmeans_delete(kmeans);
		kmeans = NULL;
	}
}

void ikm()
{
	const vl_size data_dim = 2;
	const vl_size num_data = 1000;
	const vl_size num_clusters = 3;
	const vl_size num_max_iterations = 100;

	vl_uint8 data[data_dim * num_data] = { 0, };
	for (vl_size i = 0; i < data_dim * num_data; ++i)
		data[i] = (vl_uint8)std::floor((((float)std::rand() / RAND_MAX) * 255) + 0.5f);

	//
	std::cout << "start processing ..." << std::endl;

	const VlIKMAlgorithms algorithm = VL_IKM_LLOYD;
	VlIKMFilt *ikm = vl_ikm_new(algorithm);

	//
	vl_ikm_init_rand(ikm, data_dim, num_clusters);
	//vl_ikm_init_rand_data(ikm, data, data_dim, num_data, num_clusters);

	vl_ikm_set_max_niters(ikm, num_max_iterations);

	if (-1 != vl_ikm_train(ikm, data, num_data))
	{
		const vl_ikm_acc *centers = vl_ikm_get_centers(ikm);
		for (int i = 0; i < num_data; ++i)
		{
			std::cout << '(';
			for (int j = 0; j < data_dim; ++j)
			{
				if (j) std::cout << ',';
				std::cout << data[i * data_dim + j];
			}
			std::cout << ')' << std::endl;
		}

		//
		vl_uint assignments[num_data] = { 0, };
		vl_ikm_push(ikm, assignments, data, num_data);

		vl_uint8 test_sample[data_dim] = { 0, };
		for (vl_size i = 0; i < data_dim; ++i)
			data[i] = (vl_uint8)std::floor((((float)std::rand() / RAND_MAX) * 255) + 0.5f);
		const vl_uint assignment = vl_ikm_push_one(centers, test_sample, data_dim, num_clusters);
	}
	else
	{
		std::cerr << "an overflow may have occurred." << std::endl;
	}

	std::cout << "end processing ..." << std::endl;

	//
	if (ikm)
	{
		vl_ikm_delete(ikm);
		ikm = NULL;
	}
}

void hikm()
{
	const vl_size data_dim = 2;
	const vl_size num_data = 1000;
	const vl_size num_clusters = 3;  // number of clusters per node
	const vl_size tree_depth = 5;
	const vl_size num_max_iterations = 100;

	vl_uint8 data[data_dim * num_data] = { 0, };
	for (vl_size i = 0; i < data_dim * num_data; ++i)
		data[i] = (vl_uint8)std::floor((((float)std::rand() / RAND_MAX) * 255) + 0.5f);

	//
	std::cout << "start processing ..." << std::endl;

	const VlIKMAlgorithms algorithm = VL_IKM_LLOYD;
	VlHIKMTree *hikm = vl_hikm_new(algorithm);

	//
	vl_hikm_init(hikm, data_dim, num_clusters, tree_depth);
	vl_hikm_set_max_niters(hikm, num_max_iterations);

	vl_hikm_train(hikm, data, num_data);
	{
		const VlHIKMNode *root = vl_hikm_get_root(hikm);

		// TODO [implement] >>

		//
		vl_uint assignments[num_data] = { 0, };
		vl_hikm_push(hikm, assignments, data, num_data);
	}

	std::cout << "end processing ..." << std::endl;

	//
	if (hikm)
	{
		vl_hikm_delete(hikm);
		hikm = NULL;
	}
}

}  // namespace my_vlfeat
