//#include "stdafx.h"
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/io/streaming/StreamingAsciiFile.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DenseSubsetFeatures.h>
#include <shogun/base/init.h>
#include <shogun/multiclass/ShareBoost.h>
#include <iostream>
#include <algorithm>
#include <string>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/classifier_multiclass_shareboost.cpp
void classifier_multiclass_shareboost_example()
{
	const std::string fname_train("./machine_learning_data/shogun/7class_example4_train.dense");
	shogun::CStreamingAsciiFile *train_file = new shogun::CStreamingAsciiFile(fname_train.c_str());
	SG_REF(train_file);

	shogun::CStreamingDenseFeatures<float64_t> *stream_features = new shogun::CStreamingDenseFeatures<float64_t>(train_file, true, 1024);
	SG_REF(stream_features);

	shogun::SGMatrix<float64_t> mat;
	shogun::SGVector<float64_t> labvec(1000);

	stream_features->start_parser();
	shogun::SGVector<float64_t> vec;
	int32_t num_vectors = 0;
	int32_t num_feats = 0;
	while (stream_features->get_next_example())
	{
		vec = stream_features->get_vector();
		if (0 == num_feats)
		{
			num_feats = vec.vlen;
			mat = shogun::SGMatrix<float64_t>(num_feats, 1000);
		}
		std::copy(vec.vector, vec.vector + vec.vlen, mat.get_column_vector(num_vectors));
		labvec[num_vectors] = stream_features->get_label();
		++num_vectors;
		stream_features->release_example();
	}
	stream_features->end_parser();
	mat.num_cols = num_vectors;
	labvec.vlen = num_vectors;

	shogun::CMulticlassLabels *labels = new shogun::CMulticlassLabels(labvec);
	SG_REF(labels);

	// Create features with the useful values from mat
	shogun::CDenseFeatures<float64_t> *features = new shogun::CDenseFeatures<float64_t>(mat);
	SG_REF(features);

	SG_SPRINT("Performing ShareBoost on a %d-class problem\n", labels->get_num_classes());

	// Create ShareBoost Machine
	shogun::CShareBoost *machine = new shogun::CShareBoost(features, labels, 10);
	SG_REF(machine);

	machine->train();

	shogun::SGVector<int32_t> activeset = machine->get_activeset();
	SG_SPRINT("%d out of %d features are selected:\n", activeset.vlen, mat.num_rows);
	for (int32_t i = 0; i < activeset.vlen; ++i)
		SG_SPRINT("activeset[%02d] = %d\n", i, activeset[i]);

	shogun::CDenseSubsetFeatures<float64_t> *subset_fea = new shogun::CDenseSubsetFeatures<float64_t>(features, machine->get_activeset());
	SG_REF(subset_fea);
	shogun::CMulticlassLabels *output = CMulticlassLabels::obtain_from_generic(machine->apply(subset_fea));

	int32_t correct = 0;
	for (int32_t i = 0; i < output->get_num_labels(); ++i)
		if (output->get_int_label(i) == labels->get_int_label(i))
			++correct;
	SG_SPRINT("Accuracy = %.4f\n", float64_t(correct) / labels->get_num_labels());

	// Free resources
	SG_UNREF(machine);
	SG_UNREF(output);
	SG_UNREF(subset_fea);
	SG_UNREF(features);
	SG_UNREF(labels);
	SG_UNREF(train_file);
	//SG_UNREF(stream_features);  // run-time error
}

}  // namespace my_shogun
