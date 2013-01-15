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

void kmeans()
{
	const vl_size data_dim = 2;
	const vl_size num_data = 1000;

	float data[data_dim * num_data] = { 0.0f, };
	for (vl_size i = 0; i < data_dim * num_data; ++i)
		data[i] = (float)std::rand() / RAND_MAX;

	//
	std::cout << "start processing ..." << std::endl;

	const vl_size num_clusters = 3;
	const VlVectorComparisonType distance = VlDistanceL2;
	const vl_size num_max_iterations = 100;

	VlKMeans *kmeans = vl_kmeans_new(VL_TYPE_DOUBLE, distance);

	vl_kmeans_set_max_num_iterations(kmeans, num_max_iterations);

	if (true)
	{
		// initialization
#if 1
		vl_kmeans_seed_centers_with_rand_data(kmeans, (void const *)data, data_dim, num_data, num_clusters);
#elif 0
		vl_kmeans_seed_centers_plus_plus(kmeans, (void const *)data, data_dim, num_data, num_clusters);
#else
		{
			float init_centers[data_dim * num_clusters] = { 0.0f, };
			for (vl_size i = 0; i < data_dim * num_clusters; ++i)
				init_centers[i] = (float)std::rand() / RAND_MAX;
			vl_kmeans_set_centers(kmeans, (void const *)init_centers, data_dim, num_clusters);
		}
#endif

		// clustering
		const double energy = vl_kmeans_refine_centers(kmeans, data, num_data);
	}
	else
	{
		const VlKMeansAlgorithm algorithm = VlKMeansLloyd;  // VlKMeansLloyd, VlKMeansElkan, VlKMeansANN
		const VlKMeansInitialization initialization = VlKMeansRandomSelection;  // VlKMeansRandomSelection, VlKMeansPlusPlus
		const vl_size num_repetitions = 100;

		vl_kmeans_set_algorithm(kmeans, VlKMeansLloyd);
		vl_kmeans_set_initialization(kmeans, VlKMeansRandomSelection);
		vl_kmeans_set_num_repetitions(kmeans, num_repetitions);

		// clustering
		const double energy = vl_kmeans_cluster(kmeans, (void const *)data, data_dim, num_data, num_clusters);
	}

	//
	const vl_size num_iterations = vl_kmeans_get_num_repetitions(kmeans);
	//const vl_type data_type = vl_kmeans_get_data_type(kmeans);

	//
	{
		const float *centers = (float *)vl_kmeans_get_centers(kmeans);
		for (int i = 0; i < num_clusters; ++i)
		{
			std::cout << '(';
			for (int j = 0; j < data_dim; ++j)
			{
				if (j) std::cout << ',';
				std::cout << centers[i * data_dim + j];
			}
			std::cout << ')' << std::endl;
		}
	}

	//
	{
		vl_uint32 assignments[num_data] = { 0, };
		double distances[num_data] = { 0, };
		vl_kmeans_quantize(kmeans, assignments, (void *)distances, (void const *)data, num_data);

		for (int i = 0; i < num_data; ++i)
		{
			std::cout << '(';
			for (int j = 0; j < data_dim; ++j)
			{
				if (j) std::cout << ',';
				std::cout << data[i * data_dim + j];  // TODO [check] >> is it correct?
			}
			std::cout << ") => " << assignments[i] << std::endl;
		}
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

	vl_uint8 data[data_dim * num_data] = { 0, };
	for (vl_size i = 0; i < data_dim * num_data; ++i)
		data[i] = (vl_uint8)std::floor((((float)std::rand() / RAND_MAX) * 255) + 0.5f);

	//
	std::cout << "start processing ..." << std::endl;

	const vl_size num_clusters = 3;
	const VlIKMAlgorithms algorithm = VL_IKM_LLOYD;  // VL_IKM_LLOYD, VL_IKM_ELKAN
	const vl_size num_max_iterations = 100;

	VlIKMFilt *ikm = vl_ikm_new(algorithm);

	vl_ikm_set_max_niters(ikm, num_max_iterations);

	// initialization
#if 1
	vl_ikm_init_rand(ikm, data_dim, num_clusters);
#elif 0
	vl_ikm_init_rand_data(ikm, data, data_dim, num_data, num_clusters);
#else
	vl_ikm_acc init_centers[data_dim * num_clusters] = { 0, };
	for (vl_size i = 0; i < data_dim * num_clusters; ++i)
		init_centers[i] = (vl_ikm_acc)std::floor((((float)std::rand() / RAND_MAX) * 255) + 0.5f);
	vl_ikm_init(ikm, init_centers, data_dim, num_clusters);
#endif

	// training
	if (-1 != vl_ikm_train(ikm, data, num_data))
	{
		const vl_ikm_acc *centers = vl_ikm_get_centers(ikm);
		for (int i = 0; i < num_clusters; ++i)
		{
			std::cout << '(';
			for (int j = 0; j < data_dim; ++j)
			{
				if (j) std::cout << ',';
				std::cout << (int)centers[i * data_dim + j];
			}
			std::cout << ')' << std::endl;
		}

		//
		{
			vl_uint assignments[num_data] = { 0, };
			vl_ikm_push(ikm, assignments, data, num_data);
			for (int i = 0; i < num_data; ++i)
			{
				std::cout << '(';
				for (int j = 0; j < data_dim; ++j)
				{
					if (j) std::cout << ',';
					std::cout << (int)data[i * data_dim + j];  // TODO [check] >> is it correct?
				}
				std::cout << ") => " << assignments[i] << std::endl;
			}
		}

		// testing
		{
			vl_uint8 test_sample[data_dim] = { 0, };
			for (vl_size i = 0; i < data_dim; ++i)
				test_sample[i] = (vl_uint8)std::floor((((float)std::rand() / RAND_MAX) * 255) + 0.5f);
			const vl_uint assignment = vl_ikm_push_one(centers, test_sample, data_dim, num_clusters);
		}
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

	vl_uint8 data[data_dim * num_data] = { 0, };
	for (vl_size i = 0; i < data_dim * num_data; ++i)
		data[i] = (vl_uint8)std::floor((((float)std::rand() / RAND_MAX) * 255) + 0.5f);

	//
	std::cout << "start processing ..." << std::endl;

	const VlIKMAlgorithms algorithm = VL_IKM_LLOYD;
	const vl_size num_clusters = 3;  // number of clusters per node
	const vl_size tree_depth = 5;
	const vl_size num_max_iterations = 100;

	VlHIKMTree *hikm = vl_hikm_new(algorithm);

	vl_hikm_set_max_niters(hikm, num_max_iterations);

	// initilization
	vl_hikm_init(hikm, data_dim, num_clusters, tree_depth);

	// training
	vl_hikm_train(hikm, data, num_data);
	{
		//const int ndims = vl_hikm_get_ndims(hikm);  // dim. of data
		//const int K = vl_hikm_get_K(hikm);  // num. of clusters
		//const int depth = vl_hikm_get_depth(hikm);  // tree depth
		//std::cout << "ndims: " << ndims << ", K: " << K << ", depth: " << depth << std::endl;

		const VlHIKMNode *root = vl_hikm_get_root(hikm);

		// TODO [implement] >> visualize the tree

		//
		{
			vl_uint assignments[num_data * tree_depth] = { 0, };
			vl_hikm_push(hikm, assignments, data, num_data);
			for (int i = 0; i < num_data; ++i)
			{
				std::cout << '(';
				for (int j = 0; j < data_dim; ++j)
					std::cout << (0 == j ? "" : ",") << (int)data[i * data_dim + j];  // TODO [check] >> is it correct?
				std::cout << ") => ";
				for (int k = 0; k < tree_depth; ++k)
					std::cout << (0 == k ? "" : "-") << assignments[i * tree_depth + k];  // TODO [check] >> is it correct?
				std::cout << std::endl;
			}
		}
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
