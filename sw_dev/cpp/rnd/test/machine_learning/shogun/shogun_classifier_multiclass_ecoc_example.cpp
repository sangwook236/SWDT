//#include "stdafx.h"
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/io/streaming/StreamingAsciiFile.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/multiclass/ecoc/ECOCStrategy.h>
#include <shogun/multiclass/ecoc/ECOCOVREncoder.h>
#include <shogun/multiclass/ecoc/ECOCHDDecoder.h>
#include <shogun/machine/LinearMulticlassMachine.h>
#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/base/init.h>


#define  EPSILON  1e-5

namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/classifier_multiclass_ecoc.cpp
void classifier_multiclass_ecoc_example()
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
	shogun::CDenseFeatures<float64_t> *features =(shogun::CDenseFeatures<float64_t> *)stream_features->get_streamed_features(1000);

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

	// Create liblinear svm classifier with L2-regularized L2-loss
	shogun::CLibLinear *svm = new shogun::CLibLinear(L2R_L2LOSS_SVC);
	SG_REF(svm);

	// Add some configuration to the svm
	svm->set_epsilon(EPSILON);
	svm->set_bias_enabled(true);

	// Create a multiclass svm classifier that consists of several of the previous one
	shogun::CLinearMulticlassMachine *mc_svm = new shogun::CLinearMulticlassMachine(new shogun::CECOCStrategy(new shogun::CECOCOVREncoder(), new shogun::CECOCHDDecoder()), (shogun::CDotFeatures *)features, svm, labels);
	SG_REF(mc_svm);

	// Train the multiclass machine using the data passed in the constructor
	mc_svm->train();

	// Classify the training examples and show the results
	shogun::CMulticlassLabels *output = shogun::CMulticlassLabels::obtain_from_generic(mc_svm->apply());

	shogun::SGVector<int32_t> out_labels = output->get_int_labels();
	shogun::SGVector<int32_t>::display_vector(out_labels.vector, out_labels.vlen);

	// Free resources
	SG_UNREF(mc_svm);
	SG_UNREF(svm);
	SG_UNREF(output);
	SG_UNREF(features);
	SG_UNREF(labels);
	SG_UNREF(ffeats_train);
	SG_UNREF(flabels_train);
	SG_UNREF(stream_features);
	SG_UNREF(stream_labels);
}

}  // namespace my_shogun
