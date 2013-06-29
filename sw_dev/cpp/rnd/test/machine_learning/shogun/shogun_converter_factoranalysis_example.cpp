//#include "stdafx.h"
#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/converter/FactorAnalysis.h>
#include <shogun/mathematics/Math.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/converter_factoranalysis.cpp
void converter_factoranalysis_example()
{
	const int N = 100;
	const int dim = 3;
	
	float64_t *matrix = new double[N * dim];
	for (int i = 0; i < N * dim; ++i)
		matrix[i] = shogun::CMath::sin((i / float64_t(N * dim)) * 3.14);

	shogun::CDenseFeatures<double> *features = new shogun::CDenseFeatures<double>(shogun::SGMatrix<double>(matrix, dim, N));
	SG_REF(features);
	
	shogun::CFactorAnalysis *fa = new shogun::CFactorAnalysis();
	shogun::CDenseFeatures<double> *embedding = fa->embed(features);
	
	SG_UNREF(embedding);
	SG_UNREF(fa);
	SG_UNREF(features);
}

}  // namespace my_shogun
