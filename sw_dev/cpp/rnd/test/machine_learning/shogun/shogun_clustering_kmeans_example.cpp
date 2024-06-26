//#include "stdafx.h"
#include <shogun/base/init.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/modelselection/GridSearchModelSelection.h>
#include <shogun/modelselection/ModelSelectionParameters.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/clustering/KMeans.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/distance/MinkowskiMetric.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/clustering_kmeans.cpp
void clustering_kmeans_example()
{
	const int32_t num_clusters = 4;
	const int32_t num_features = 11;  // the number of labels (?)
	const int32_t dim_features = 3;
	const int32_t num_vectors_per_cluster = 5;
	const float64_t cluster_std_dev = 2.0;

	// build random cluster centers
	shogun::SGMatrix<float64_t> cluster_centers(dim_features, num_clusters);
	shogun::SGVector<float64_t>::random_vector(cluster_centers.matrix, dim_features * num_clusters, -10.0, 10.0);
	shogun::SGMatrix<float64_t>::display_matrix(cluster_centers.matrix, cluster_centers.num_rows, cluster_centers.num_cols, "cluster centers");

	// create data around clusters
	shogun::SGMatrix<float64_t> data(dim_features, num_clusters * num_vectors_per_cluster);
	for (index_t i = 0; i < num_clusters; ++i)
	{
		for (index_t j = 0; j < dim_features; ++j)
		{
			for (index_t k = 0; k < num_vectors_per_cluster; ++k)
			{
				index_t idx = i * dim_features * num_vectors_per_cluster;
				idx += j;
				idx += k * dim_features;
				const float64_t entry = cluster_centers.matrix[i * dim_features + j];
				data.matrix[idx] = shogun::CMath::normal_random(entry, cluster_std_dev);
			}
		}
	}

	// create features, SG_REF to avoid deletion
	shogun::CDenseFeatures<float64_t> *features = new shogun::CDenseFeatures<float64_t>();
	features->set_feature_matrix(data);
	SG_REF(features);

	// create labels for cluster centers
	shogun::CMulticlassLabels *labels = new shogun::CMulticlassLabels(num_features);
	for (index_t i = 0; i < num_features; ++i)
		labels->set_label(i, i % 2 == 0 ? 0 : 1);

	// create distance
	shogun::CEuclideanDistance *distance = new shogun::CEuclideanDistance(features, features);

	// create distance machine
	shogun::CKMeans *clustering = new shogun::CKMeans(num_clusters, distance);
	clustering->train(features);

	// build clusters
/*
	shogun::CMulticlassLabels *result = shogun::CMulticlassLabels::obtain_from_generic(clustering->apply());
	for (index_t i = 0; i < result->get_num_labels(); ++i)
		SG_SPRINT("cluster index of vector %i: %f\n", i, result->get_label(i));
*/

	// print cluster centers
	shogun::CDenseFeatures<float64_t> *centers = (shogun::CDenseFeatures<float64_t> *)distance->get_lhs();

	shogun::SGMatrix<float64_t> centers_matrix = centers->get_feature_matrix();

	shogun::SGMatrix<float64_t>::display_matrix(centers_matrix.matrix, centers_matrix.num_rows, centers_matrix.num_cols, "learned centers");
	shogun::SGMatrix<float64_t>::display_matrix(cluster_centers.matrix, cluster_centers.num_rows, cluster_centers.num_cols, "real centers");

	// clean up
	//SG_UNREF(result);
	SG_UNREF(centers);
	SG_UNREF(clustering);
	SG_UNREF(labels);
	SG_UNREF(features);
}

}  // namespace my_shogun
