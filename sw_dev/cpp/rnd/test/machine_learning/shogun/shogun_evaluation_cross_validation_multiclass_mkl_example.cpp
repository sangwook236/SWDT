//#include "stdafx.h"
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/io/streaming/StreamingAsciiFile.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/kernel/PolyKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/classifier/mkl/MKLMulticlass.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/MulticlassAccuracy.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/evaluation_cross_validation_multiclass_mkl.cpp
void evaluation_cross_validation_multiclass_mkl_example()
{
	// init random number generator for reproducible results of cross-validation in the light of ASSERT(result->mean>0.9); some lines down below
	shogun::CMath::init_random(1);

	// stream data from a file
	const int32_t num_vectors = 50;
	const int32_t num_feats = 2;

	// file data
	const char fname_feats[] = "../data/fm_train_real.dat";
	const char fname_labels[] = "../data/label_train_multiclass.dat";
	shogun::CStreamingAsciiFile *ffeats_train = new shogun::CStreamingAsciiFile(fname_feats);
	shogun::CStreamingAsciiFile *flabels_train = new shogun::CStreamingAsciiFile(fname_labels);
	SG_REF(ffeats_train);
	SG_REF(flabels_train);

	// streaming data
	shogun::CStreamingDenseFeatures<float64_t> *stream_features = new shogun::CStreamingDenseFeatures<float64_t>(ffeats_train, false, 1024);
	shogun::CStreamingDenseFeatures<float64_t> *stream_labels = new shogun::CStreamingDenseFeatures<float64_t>(flabels_train, true, 1024);
	SG_REF(stream_features);
	SG_REF(stream_labels);

	// matrix data
	shogun::SGMatrix<float64_t> mat = shogun::SGMatrix<float64_t>(num_feats, num_vectors);
	shogun::SGVector<float64_t> vec;
	stream_features->start_parser();

	index_t count = 0;
	while (stream_features->get_next_example() && count < num_vectors)
	{
		vec = stream_features->get_vector();
		for (int32_t i = 0; i < num_feats; ++i)
			mat(i, count) = vec[i];

		stream_features->release_example();
		++count;
	}
	stream_features->end_parser();
	mat.num_cols = num_vectors;

	// dense features from streamed matrix
	shogun::CDenseFeatures<float64_t> *features = new shogun::CDenseFeatures<float64_t>(mat);
	shogun::CMulticlassLabels *labels = new shogun::CMulticlassLabels(num_vectors);
	SG_REF(features);
	SG_REF(labels);

	// read labels from file
	int32_t idx = 0;
	stream_labels->start_parser();
	while (stream_labels->get_next_example())
	{
		labels->set_int_label(idx++, (int32_t)stream_labels->get_label());
		stream_labels->release_example();
	}
	stream_labels->end_parser();

	// combined features and kernel
	CCombinedFeatures *cfeats=new CCombinedFeatures();
	CCombinedKernel *cker=new CCombinedKernel();
	SG_REF(cfeats);
	SG_REF(cker);

	// 1st kernel: gaussian
	cfeats->append_feature_obj(features);
	cker->append_kernel(new shogun::CGaussianKernel(features, features, 1.2, 10));

	// 2nd kernel: linear
	cfeats->append_feature_obj(features);
	cker->append_kernel(new shogun::CLinearKernel(features, features));

	// 3rd kernel: poly
	cfeats->append_feature_obj(features);
	cker->append_kernel(new shogun::CPolyKernel(features, features, 2, true, 10));

	cker->init(cfeats, cfeats);

	// create mkl instance
	shogun::CMKLMulticlass *mkl = new shogun::CMKLMulticlass(1.2, cker, labels);
	SG_REF(mkl);
	mkl->set_epsilon(0.00001);
	mkl->parallel->set_num_threads(1);
	mkl->set_mkl_epsilon(0.001);
	mkl->set_mkl_norm(1.5);

	// train to see weights
	mkl->train();
	cker->get_subkernel_weights().display_vector("weights");

	// cross-validation instances
	const index_t n_folds = 3;
	const index_t n_runs = 5;
	shogun::CMulticlassAccuracy *eval_crit = new CMulticlassAccuracy();
	shogun::CStratifiedCrossValidationSplitting *splitting = new CStratifiedCrossValidationSplitting(labels, n_folds);
	shogun::CCrossValidation *cross = new CCrossValidation(mkl, cfeats, labels, splitting, eval_crit);
	cross->set_autolock(false);
	cross->set_num_runs(n_runs);
	cross->set_conf_int_alpha(0.05);

	// perform x-val and print result
	shogun::CCrossValidationResult *result = (shogun::CCrossValidationResult *)cross->evaluate();
	SG_SPRINT("mean of %d %d-fold x-val runs: %f\n", n_runs, n_folds, result->mean);

	// assert high accuracy
	ASSERT(result->mean>0.9);

	// clean up
	SG_UNREF(ffeats_train);
	SG_UNREF(flabels_train);
	SG_UNREF(stream_features);
	SG_UNREF(stream_labels);
	SG_UNREF(features);
	SG_UNREF(labels);
	SG_UNREF(cfeats);
	SG_UNREF(cker);
	SG_UNREF(mkl);
	SG_UNREF(cross);
	SG_UNREF(result);
}

}  // namespace my_shogun
