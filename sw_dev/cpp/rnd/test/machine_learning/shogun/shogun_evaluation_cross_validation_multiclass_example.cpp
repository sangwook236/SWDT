//#include "stdafx.h"
#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/MulticlassLibLinear.h>
#include <shogun/io/streaming/StreamingAsciiFile.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/evaluation/MulticlassAccuracy.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/evaluation_cross_validation_multiclass.cpp
void evaluation_cross_validation_multiclass_example()
{
	// Prepare to read a file for the training data
	const char fname_feats[] = "../data/fm_train_real.dat";
	const char fname_labels[] = "../data/label_train_multiclass.dat";
	shogun::CStreamingAsciiFile *ffeats_train = new shogun::CStreamingAsciiFile(fname_feats);
	shogun::CStreamingAsciiFile *flabels_train = new shogun::CStreamingAsciiFile(fname_labels);
	SG_REF(ffeats_train);
	SG_REF(flabels_train);

	shogun::CStreamingDenseFeatures<float64_t> *stream_features = new shogun::CStreamingDenseFeatures<float64_t>(ffeats_train, false, 1024);
	shogun::CStreamingDenseFeatures<float64_t> *stream_labels = new shogun::CStreamingDenseFeatures<float64_t>(flabels_train, true, 1024);
	SG_REF(stream_features);
	SG_REF(stream_labels);

	stream_features->start_parser();

	// Read the values from the file and store them in features
	shogun::CDenseFeatures<float64_t> *features = (shogun::CDenseFeatures<float64_t> *)stream_features->get_streamed_features(1000);

	stream_features->end_parser();

	shogun::CMulticlassLabels *labels = new shogun::CMulticlassLabels(features->get_num_vectors());
	SG_REF(features);
	SG_REF(labels);

	// Read the labels from the file
	int32_t idx = 0;
	stream_labels->start_parser();
	while (stream_labels->get_next_example())
	{
		labels->set_int_label(idx++, (int32_t)stream_labels->get_label());
		stream_labels->release_example();
	}
	stream_labels->end_parser();

	// create svm via libsvm
	const float64_t svm_C = 10;
	const float64_t svm_eps = 0.0001;
	shogun::CMulticlassLibLinear *svm = new shogun::CMulticlassLibLinear(svm_C, features, labels);
	svm->set_epsilon(svm_eps);

	// train and output
	svm->train(features);
	shogun::CMulticlassLabels *output = shogun::CMulticlassLabels::obtain_from_generic(svm->apply(features));
	for (index_t i = 0; i < features->get_num_vectors(); ++i)
		SG_SPRINT("i = %d, class = %f,\n", i, output->get_label(i));

	// evaluation criterion
	shogun::CMulticlassAccuracy *eval_crit = new shogun::CMulticlassAccuracy ();

	// evaluate training error
	const float64_t eval_result = eval_crit->evaluate(output, labels);
	SG_SPRINT("training accuracy: %f\n", eval_result);
	SG_UNREF(output);

	// assert that regression "works".
	// this is not guaranteed to always work but should be a really coarse check to see if everything is going approx. right
	ASSERT(eval_result<2);

	// splitting strategy
	const index_t n_folds = 5;
	shogun::CStratifiedCrossValidationSplitting *splitting = new shogun::CStratifiedCrossValidationSplitting(labels, n_folds);

	// cross validation instance, 10 runs, 95% confidence interval
	shogun::CCrossValidation *cross = new CCrossValidation(svm, features, labels, splitting, eval_crit);

	cross->set_num_runs(1);
	cross->set_conf_int_alpha(0.05);

	// actual evaluation
	shogun::CCrossValidationResult *result=(shogun::CCrossValidationResult *)cross->evaluate();

	if (result->get_result_type() != CROSSVALIDATION_RESULT)
		SG_SERROR("Evaluation result is not of type CCrossValidationResult!");

	result->print_result();

	// clean up
	SG_UNREF(result);
	SG_UNREF(cross);
	SG_UNREF(features);
	SG_UNREF(labels);
	SG_UNREF(ffeats_train);
	SG_UNREF(flabels_train);
	SG_UNREF(stream_features);
	SG_UNREF(stream_labels);
}

}  // namespace my_shogun
