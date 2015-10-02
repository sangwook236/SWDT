//#include "stdafx.h"
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/DenseFeatures.h>
//#include <shogun/multiclass/ConjugateIndex.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/classifier_conjugateindex.cpp
void classifier_conjugateindex_example()
{
	// create some data
	shogun::SGMatrix<float64_t> matrix(2, 3);
	for (int32_t i = 0; i < 6; ++i)
		matrix.matrix[i] = i;

	// create three 2-dimensional vectors
	// shogun will now own the matrix created
	shogun::CDenseFeatures<float64_t> *features = new shogun::CDenseFeatures<float64_t>(matrix);

	// create three labels
	shogun::CMulticlassLabels* labels = new shogun::CMulticlassLabels(3);
	labels->set_label(0, 0);
	labels->set_label(1, +1);
	labels->set_label(2, 0);

/*
	shogun::CConjugateIndex *ci = new shogun::CConjugateIndex(features, labels);
	ci->train();

	// classify on training examples
	for (int32_t i = 0; i < 3; ++i)
		SG_SPRINT("output[%d] = %f\n", i, ci->apply_one(i));

	// free up memory
	SG_UNREF(ci);
*/
}

}  // namespace my_shogun
