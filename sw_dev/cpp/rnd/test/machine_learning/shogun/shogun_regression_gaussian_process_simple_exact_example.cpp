//#include "stdafx.h"
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/base/init.h>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_shogun {

using namespace shogun;

// [ref] ${SHOGUN_HOME}/examples/undocumented/libshogun/regression_gaussian_process_simple_exact.cpp
void regression_gaussian_process_simple_exact_example()
{
	// create some easy regression data: 1d noisy sine wave
	const index_t n = 100;
	const float64_t x_range = 6;

	shogun::SGMatrix<float64_t> X(1, n);
	shogun::SGMatrix<float64_t> X_test(1, n);
	shogun::SGVector<float64_t> Y(n);

	for (index_t i = 0; i < n; ++i)
	{
		X[i] = shogun::CMath::random(0.0, x_range);
		X_test[i] = (float64_t)i / n * x_range;
		Y[i] = shogun::CMath::sin(X[i]);
	}

	// shogun representation
	shogun::CDenseFeatures<float64_t> *feat_train = new shogun::CDenseFeatures<float64_t>(X);
	shogun::CDenseFeatures<float64_t> *feat_test = new shogun::CDenseFeatures<float64_t>(X_test);
	shogun::CRegressionLabels *label_train = new shogun::CRegressionLabels(Y);

	// specity GPR with exact inference
	const float64_t sigma = 1;
	const float64_t shogun_sigma = sigma * sigma * 2;
	shogun::CGaussianKernel *kernel = new shogun::CGaussianKernel(10, shogun_sigma);
	shogun::CZeroMean *mean = new shogun::CZeroMean();
	shogun::CGaussianLikelihood *lik = new shogun::CGaussianLikelihood();
	lik->set_sigma(1);

	shogun::CExactInferenceMethod *inf = new shogun::CExactInferenceMethod(kernel, feat_train, mean, label_train, lik);
	shogun::CFeatures *latent = inf->get_latent_features();
	SG_UNREF(latent);
	shogun::CGaussianProcessRegression *gpr = new shogun::CGaussianProcessRegression(inf, feat_train, label_train);

	// perform inference
	gpr->set_return_type(shogun::CGaussianProcessRegression::GP_RETURN_MEANS);
	shogun::CRegressionLabels *predictions = gpr->apply_regression(feat_test);
	predictions->get_labels().display_vector("predictions");

	SG_UNREF(predictions);
	SG_UNREF(gpr);
}

}  // namespace my_shogun
