//#include "stdafx.h"
// MCMC++ includes.
#include <mcmc++/Density.h>
#include <mcmc++/MCMC.h>
#include <mcmc++/intervals.h>
#include <mcmc++/statistics.h>
#include <mcmc++/util.h>
// Boost includes.
#include <boost/any.hpp>
#include <boost/format.hpp>
// standard includes.
#include <iostream>
#include <string>
#include <vector>
#include <cmath>


namespace {
namespace local {

lot rng_(lot::RAN_MT, lot::ZERO);

#if defined(METRO)
typedef MetroStep StepType;
#elif defined(DIRECT)
typedef FunctionStep StepType;
#else
typedef SliceStep StepType;
#endif

class BinomialModel;

class p : public Parameter
{
public:
	p(BinomialModel *bin)
	: Parameter("p"), bin_(bin)
	{
		Assign(0.5);
	}

public:
	// likelihood associated with current parameter.
	/*virtual*/ double llike(const double p0) const;

	// propose a new value for a parameter in an M-H step.
	/*virtual*/ double propose(const double current) const
	{
		return proposeBeta(current, 5, Util::dbl_eps);
	}

	// probability of proposing x, given starting from y.
	/*virtual*/ double lQ(const double x, const double y) const
	{
		return logQBeta(x, y, 5, Util::dbl_eps);
	}

	// function used to update a deterministic node.
	/*virtual*/ const double Function(const bool doCalc = true) const;

private:
	BinomialModel *bin_;
};

class BinomialModel : public Model
{
public:
	BinomialModel(const int nBurnin, const int nSample, const int thin, const int n, const int k)
	: Model(nBurnin, nSample, thin, true, false), n_(n), k_(k)
	{
		step_.push_back(new StepType(new p(this)));

		step_[0]->SetBounds(Util::dbl_min, 1.0 - Util::dbl_eps);
	}

public:
	void Report(std::ostream &outf)
	{
		const double mean = static_cast<double>(k_ + 1) / static_cast<double>(n_ + 2);
		const double variance = mean * (1.0 - mean) / (n_ + 2 + 1);
#if defined(METRO)
		outf << "Metropolis-Hastings sampler...";
#elif defined(DIRECT)
		outf << "Direct sampler...";
#else
		outf << "Slice sampler...";
#endif
		outf << std::endl << boost::format("Predicted: %|10| %|10|") % mean % std::sqrt(variance) << std::endl;

		Model::Report(outf);
	}

	inline int K(void) const
	{
		return k_;
	}

	inline int N(void) const
	{
		return n_;
	}

	// log likelihood.
	/*virtual*/ double Llike(const SampleVector &p0) const
	{
		const double p = boost::any_cast<double>(p0[0]);  // p0[0] == p.
		return Density::dbinom(k_, n_, p, true);
	}

private:
	const int n_, k_;
};

double p::llike(const double p) const
{
	return Density::dbinom(bin_->K(), bin_->N(), p, true);
}

const double p::Function(const bool doCalc) const
{
	return rng_.beta(bin_->K() + 1, bin_->N() - bin_->K() + 1);
}

}  // namespace local
}  // unnamed namespace

namespace my_mcmcpp {

// [ref] ${MCMC++_HOME}/examples/binomial.cpp.
void binomial_example()
{
	// density of the binomial distribution returns probability of getting k successes in n binomial trials with a probability p of success on each trial.
	const int n = 10;
	const int k = 7;

	const int nBurnin = 1000;
	const int nSample = 10000;
	const int thin = 5;

	local::BinomialModel model(nBurnin, nSample, thin, n, k);
	std::cout << "start simulation ..." << std::endl;
	model.Simulation(std::cerr, false);
	std::cout << "end simulation ..." << std::endl;

	//
	model.Report(std::cout);
}

}  // namespace my_mcmcpp
