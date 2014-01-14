#include <vl/generic.h>
#include <vl/gmm.h>
#include <vl/host.h>
#include <vl/kmeans.h>
#include <vl/fisher.h>
#include <vl/vlad.h>
#include <iostream>
#include <cstdio>


namespace {
namespace local {

void saveResults(const char *dataFileData, const char *dataFileResults, VlGMM *gmm, void *data, vl_size numData)
{
	vl_size d, cIdx;
	vl_uindex i_d;

	const vl_size dimension = vl_gmm_get_dimension(gmm) ;
	const vl_size numClusters = vl_gmm_get_num_clusters(gmm) ;
	const vl_type dataType = vl_gmm_get_data_type(gmm) ;
	double const *sigmas = (double const *)vl_gmm_get_covariances(gmm) ;
	double const *means = (double const *)vl_gmm_get_means(gmm) ;
	double const *weights = (double const *)vl_gmm_get_priors(gmm) ;
	double const *posteriors = (double const *)vl_gmm_get_posteriors(gmm) ;

	const char *mode = "w";
	FILE *ofp = fopen(dataFileData, mode);
	for (i_d = 0; i_d < numData; ++i_d)
	{
		if (vl_gmm_get_data_type(gmm) == VL_TYPE_DOUBLE)
		{
			for (d = 0; d < vl_gmm_get_dimension(gmm); ++d)
			{
				fprintf(ofp, "%f ", ((double *)data)[i_d * vl_gmm_get_dimension(gmm) + d]);
			}
		}
		else
		{
			for (d = 0; d < vl_gmm_get_dimension(gmm); d++)
			{
				fprintf(ofp, "%f ", ((float *) data)[i_d * vl_gmm_get_dimension(gmm) + d]);
			}
		}
		fprintf(ofp, "\n");
	}
	fclose(ofp);

	ofp = fopen(dataFileResults, mode);
	for (cIdx = 0; cIdx < numClusters; ++cIdx)
	{
		if (VL_TYPE_DOUBLE == dataType)
		{
			for (d = 0; d < vl_gmm_get_dimension(gmm); ++d)
			{
				fprintf(ofp, "%f ", ((double *)means)[cIdx*dimension + d]);
			}
			for (d = 0; d < dimension; ++d)
			{
				fprintf(ofp, "%f ", ((double *)sigmas)[cIdx*dimension + d]);
			}
			fprintf(ofp, "%f ", ((double *)weights)[cIdx]);
			for (i_d = 0; i_d < numData; ++i_d)
			{
				fprintf(ofp, "%f ", ((double *)posteriors)[cIdx*numData + i_d]);
			}
			fprintf(ofp, "\n");
		}
		else
		{
			for (d = 0; d < dimension; ++d)
			{
				fprintf(ofp, "%f ", ((float *)means)[cIdx*dimension + d]);
			}
			for (d = 0; d < dimension; ++d)
			{
				fprintf(ofp, "%f ", ((float *)sigmas)[cIdx*dimension + d]);
			}
			fprintf(ofp, "%f ", ((float *)weights)[cIdx]);
			for (i_d = 0; i_d < numData; ++i_d)
			{
				fprintf(ofp, "%f ", ((float *)posteriors)[cIdx*numData + i_d]);
			}
			fprintf(ofp, "\n");
		}
	}
	fclose(ofp);
}

}  // namespace local
}  // unnamed namespace

namespace my_vlfeat {

#define TYPE float
#define VL_F_TYPE VL_TYPE_FLOAT

// [ref] ${VLFEAT_HOME}/src/test_gmm.c.
// [ref] http://www.vlfeat.org/api/gmm.html.
void gmm()
{
	const vl_bool computeFisher = VL_TRUE;
	const vl_bool computeVlad = VL_TRUE;

	const double sigmaLowerBound = 0.000001;

	const vl_size numData = 1000;
	const vl_size dimension = 3;
	const vl_size numClusters = 20;
	const vl_size maxiter = 5;
	const vl_size maxrep = 1;

	typedef enum _init
	{
		KMeans,
		Rand,
		Custom
	} Init;
	const Init init = KMeans;

	TYPE *data = (TYPE *)vl_malloc(sizeof(TYPE) * numData * dimension);

	vl_set_num_threads(0) ;  // use the default number of threads.

	VlRand rand;
	vl_rand_init(&rand) ;
	vl_rand_seed(&rand, 49000);

	vl_size dataIdx, d, cIdx;
	for (dataIdx = 0; dataIdx < numData; ++dataIdx)
	{
		for (d = 0; d < dimension; ++d)
		{
			data[dataIdx * dimension + d] = (TYPE)vl_rand_real3(&rand);
			//VL_PRINT("%f ", data[dataIdx * dimension + d]);
		}
		//VL_PRINT("\n");
	}

	VlGMM *gmm = vl_gmm_new(VL_F_TYPE, dimension, numClusters);

	switch (init)
	{
	case KMeans:
	{
		const vl_size maxiterKM = 5;
		const vl_size ntrees = 3;
		const vl_size maxComp = 20;

		VlKMeans *kmeans = vl_kmeans_new(VL_F_TYPE, VlDistanceL2);
		vl_kmeans_set_verbosity(kmeans, 1);
		vl_kmeans_set_max_num_iterations(kmeans, maxiterKM);
		vl_kmeans_set_max_num_comparisons(kmeans, maxComp);
		vl_kmeans_set_num_trees(kmeans, ntrees);
		vl_kmeans_set_algorithm(kmeans, VlKMeansANN);
		vl_kmeans_set_initialization(kmeans, VlKMeansRandomSelection);

		vl_gmm_set_initialization(gmm, VlGMMKMeans);
		vl_gmm_set_kmeans_init_object(gmm, kmeans);

		vl_kmeans_delete(kmeans);
		break;
	}

	case Rand:
		vl_gmm_set_initialization(gmm, VlGMMRand);
		break;

	case Custom:
	{
		TYPE *initSigmas = (TYPE *)vl_malloc(sizeof(TYPE) * numClusters * dimension);
		TYPE *initWeights = (TYPE *)vl_malloc(sizeof(TYPE) * numClusters);
		TYPE *initMeans = (TYPE *)vl_malloc(sizeof(TYPE) * numClusters * dimension);

		vl_gmm_set_initialization(gmm, VlGMMCustom);

		for (cIdx = 0; cIdx < numClusters; ++cIdx)
		{
			for (d = 0; d < dimension; ++d)
			{
				initMeans[cIdx * dimension + d] = (TYPE)vl_rand_real3(&rand);
				initSigmas[cIdx * dimension + d] = (TYPE)vl_rand_real3(&rand);
			}
			initWeights[cIdx] = (TYPE)vl_rand_real3(&rand);
		}

		vl_gmm_set_priors(gmm, initWeights);
		vl_gmm_set_covariances(gmm, initSigmas);
		vl_gmm_set_means(gmm, initMeans);

		break;
	}

	default:
		abort();
	}

	vl_gmm_set_max_num_iterations(gmm, maxiter);
	vl_gmm_set_num_repetitions(gmm, maxrep);
	vl_gmm_set_verbosity(gmm, 1);
	vl_gmm_set_covariance_lower_bound(gmm, sigmaLowerBound);

	//struct timeval t1, t2;
	//gettimeofday(&t1, NULL);

	vl_gmm_cluster(gmm, data, numData);
	//vl_gmm_em(gmm, data, numData);

	//gettimeofday(&t2, NULL);
	//VL_PRINT("elapsed vlfeat: %f s\n",(double)(t2.tv_sec - t1.tv_sec) + ((double)(t2.tv_usec - t1.tv_usec)) / 1000000.);

#if 0
	{
		// get the means, covariances, and priors of the GMM.
		const float *means = (float *)vl_gmm_get_means(gmm);
		const float *covariances = (float *)vl_gmm_get_covariances(gmm);
		const float *priors = (float *)vl_gmm_get_priors(gmm);
		// get loglikelihood of the estimated GMM.
		const double loglikelihood = vl_gmm_get_loglikelyhood(gmm);
		// get the soft assignments of the data points to each cluster.
		const float *posteriors = (float *)vl_gmm_get_posteriors(gmm);

		VL_PRINT("posterior:\n");
		for (cIdx = 0; cIdx < numClusters; ++cIdx)
		{
			for (dataIdx = 0; dataIdx < numData; ++dataIdx)
			{
				VL_PRINT("%f ", ((float *)posteriors)[cIdx * numData + dataIdx]);
			}
		  VL_PRINT("\n");
		}

		VL_PRINT("mean:\n");
		for (cIdx = 0; cIdx < numClusters; ++cIdx)
		{
			for(d = 0; d < dimension; ++d)
			{
				VL_PRINT("%f ", ((TYPE *)means)[cIdx * dimension + d]);
			}
			VL_PRINT("\n");
		}
	
		VL_PRINT("sigma:\n");
		for (cIdx = 0; cIdx < numClusters; ++cIdx)
		{
			for (d = 0; d < dimension; ++d)
			{
				VL_PRINT("%f ", ((TYPE *)covariances)[cIdx * dimension + d]);
			}
			VL_PRINT("\n");
		}
	
		VL_PRINT("w:\n");
		for (cIdx = 0; cIdx < numClusters; ++cIdx)
		{
			VL_PRINT("%f \n", ((TYPE *)weights)[cIdx]);  // priors ?
		}

		const char *dataFileResults = "./data/machine_vision/vlfeat/gmm-results.mat";
		const char *dataFileData = "./data/machine_vision/vlfeat/gmm-data.mat";
		local::saveResults(dataFileData, dataFileResults, gmm, (void *)data, numData);
	}
#endif

	vl_free(data);
	const vl_size numData2 = 2000;
	data = (TYPE *)vl_malloc(numData2 * dimension * sizeof(TYPE));
	for (dataIdx = 0; dataIdx < numData2; ++dataIdx)
	{
		for (d = 0; d < dimension; ++d)
		{
			data[dataIdx * dimension + d] = (TYPE)vl_rand_real3(&rand);
		}
	}

	if (computeFisher)
	{
		TYPE *enc = (TYPE *)vl_malloc(sizeof(TYPE) * 2 * dimension * numClusters);
		vl_fisher_encode(
			enc, VL_F_TYPE,
			vl_gmm_get_means(gmm), dimension, numClusters,
			vl_gmm_get_covariances(gmm),
			vl_gmm_get_priors(gmm),
			data, numData2,
			VL_FISHER_FLAG_IMPROVED
		);

		VL_PRINT("fisher:\n");
		for (cIdx = 0; cIdx < numClusters; ++cIdx)
		{
			for (d = 0; d < dimension * 2; ++d)
			{
				VL_PRINT("%f ", enc[cIdx * dimension * 2 + d]);
			}
			VL_PRINT("\n");
		}

		vl_free(enc);
	}

	if (computeVlad)
	{
		vl_uint32 *assign = (vl_uint32 *)vl_malloc(numData2 * numClusters * sizeof(vl_uint32));
		for (dataIdx = 0; dataIdx < numData2; ++dataIdx)
		{
			for (cIdx = 0; cIdx < numClusters; ++cIdx)
			{
				assign[cIdx * numData2 + dataIdx] = (vl_uint32)vl_rand_real3(&rand);
			}
		}

		TYPE *enc = (TYPE *)vl_malloc(sizeof(TYPE) * dimension * numClusters);
		vl_vlad_encode(
			enc, VL_F_TYPE,
			vl_gmm_get_means(gmm), dimension, numClusters,
			data, numData2,
			assign,
			0
		);

		vl_free(enc);
		vl_free(assign);
	}

	vl_gmm_delete(gmm);
	vl_free(data);
}

}  // namespace my_vlfeat
