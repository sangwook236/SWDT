//#include "stdafx.h"
#if !defined(__FUNCTION__)
//#if defined(UNICODE) || defined(_UNICODE)
//#define __FUNCTION__ L""
//#else
#define __FUNCTION__ ""
//#endif
#endif
#if !defined(__func__)
//#if defined(UNICODE) || defined(_UNICODE)
//#define __func__ L""
//#else
#define __func__ ""
//#endif
#endif

#include <mcmc.h>
#include <iostream>


namespace {
namespace local {

// Model specific options.
struct ModelOptions
{
	// Known sd of data.
	double sd;
	// Slice tuning parameter for mean
	double mean_slice_w;
};
// Global ModelOptions instance.
ModelOptions model_options;

// Structure to hold priors.
struct Priors
{
	// Prior on mean, mean.
	double mean_mu;
	// Prior on mean, standard deviation.
	double mean_sigma;
};
// Global Priors instance.
Priors priors;

// Parameter structure.
struct Parameters
{
	// Unknown mean of N(mean, sd).
	double mean;
};
// Global Parameter instance.
Parameters parameters;

// Data structure.
struct Data
{
	// Basic data.
	matrix X;
};
// Global Data instance.
Data data;

// Mean parameter for N(mean, model_options.sd) .
// The mean parameter is the parameter of interest in this model.
class MeanParameter : public Parameter<double>
{
public:
	// Default constructors that call base constructors. (REQUIRED)
	MeanParameter() : Parameter<double>() {}
	MeanParameter(bool track, const std::string &name) : Parameter<double>(track, name) {}

	// Log density at parameter value (+ constant)
	double LogDensity(double mean_value)
	{
		// The sum of std::log dnorms can be speed up by 10 fold by doing by getting rid of extra exp() and std::log() expressions and constant terms.
#ifndef FAST
		return scythe::sum(scythe::log(scythe::dnorm(data.X, mean_value, model_options.sd)))
			+ std::log(scythe::dnorm(mean_value, priors.mean_mu, priors.mean_sigma));
#else
		return sum_fast_log_dnorm(data.X, mean_value, model_options.sd)
			+ fast_log_dnorm(mean_value, priors.mean_mu, priors.mean_sigma);
#endif
	}

	// Draw starting value.
	double StartingValue()
	{
		return myrng.rnorm(priors.mean_mu, model_options.sd);
	}

	// Save back to global location.
	void Save(double new_value)
	{
		parameters.mean = new_value;
	}

	// Return value from global location.
	double Value()
	{
		return parameters.mean;
	}
};

}  // namespace local
}  // unnamed namespace

namespace my_scythemcmc {

// [ref] ${SCYTHEMCMC_HOME}/examples/normal.cpp.
void normal_example()
{
	MCMCOptions	mcmc_options;
	{
		mcmc_options.config_file = std::string();  // config file name. don't need.
		mcmc_options.out_file = "./data/statistical_analysis/scythe/mcmc_normal_output.txt";  // out file name.
		mcmc_options.chains = 1;  // number of chains.
		mcmc_options.sample_size = 1000;  // retained sample size.
		mcmc_options.burnin = 0;  // burn in period.
		mcmc_options.thin = 1;  // thinning iterval (1 = no thinning).
		// random number seed (0 uses current timestamp, 1 attempts to load from lecuyer.seed file).
		//	[ref] ShowUsage() in mcmc.h.
		mcmc_options.random_seed[0] = 0L;
		mcmc_options.random_seed[1] = 0L;
		mcmc_options.random_seed[2] = 0L;
		mcmc_options.random_seed[3] = 0L;
		mcmc_options.random_seed[4] = 0L;
		mcmc_options.random_seed[5] = 0L;
	}

	{
		std::cout << "Loading model options..." << std::endl;
		local::model_options.sd = 1.0;
		local::model_options.mean_slice_w = 1.0;

		std::cout << "Loading priors..." << std::endl;
		local::priors.mean_mu = 0.0;
		local::priors.mean_sigma = 2.0;

		// TODO [check] >> is it correct?
		scythe::Matrix<double, scythe::Col> X(3, 4, false);
		X = 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12;

		std::cout << "Loading data..." << std::endl;
		local::data.X = X;
		std::cout << "Observations added: " << local::data.X.rows() << std::endl;
	}

	//
	Sampler sampler(mcmc_options);

	std::cout << "Adding mean parameter..." << std::endl;
	// Define parameter called "mean", and track it (true).
	// See other examples for how you use loops to creating multiple instance of a specific parameter type (e.g. ideal points).
	local::MeanParameter mean_parameter(true, "mean");

	// Add sampling step.
	sampler.AddStep(new SliceStep<local::MeanParameter>(mean_parameter, local::model_options.mean_slice_w, -dInf, dInf));
	//sampler.AddStep(new MetropStep<local::MeanParameter, NormalProposal>(mean_parameter, NormalProposal(1.0)));
	//sampler.AddStep(new MetropStep<local::MeanParameter, BetaProposal>(mean_parameter, BetaProposal(1.0)));
	//sampler.AddStep(new MetropStep<local::MeanParameter, LogNormalProposal>(mean_parameter, LogNormalProposal(1.0)));
	//sampler.AddStep(new GibbsStep<local::MeanParameter, double>(mean_parameter));
	//sampler.AddStep(new FunctionStep<local::MeanParameter, double>(mean_parameter));

	sampler.Run();
}

}  // namespace my_scythemcmc
