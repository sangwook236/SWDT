//#include "stdafx.h"
// MCMC++ includes.
#include <mcmc++/DataTable.h>
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

class NormalModel;

class mean : public Parameter
{
public:
	mean(NormalModel *norm)
	: Parameter("mean"), norm_(norm)
	{
		Assign(0.0);
	}

public:
	// likelihood associated with current parameter.
	/*virtual*/ double llike(const double mu) const;
	// prior associated with current parameter.
	/*virtual*/ double lPrior(const double mu0) const;

private:
	NormalModel *norm_;
};

class variance : public Parameter
{
public:
	variance(NormalModel *norm)
	: Parameter("variance"), norm_(norm)
	{
		Assign(1.0);
	}

public:
	// likelihood associated with current parameter.
	/*virtual*/ double llike(const double var) const;
	// prior associated with current parameter.
	/*virtual*/ double lPrior(const double var0) const;

private:
	NormalModel *norm_;
};

class precision : public Parameter
{
public:
	precision(NormalModel *norm)
	: Parameter("precision"), norm_(norm)
	{}

public:
	// function used to update a deterministic node.
	/*virtual*/ const double Function(const bool doCalc = true) const;

private:
	NormalModel *norm_;
};

class NormalModel : public Model
{
public:
	NormalModel(
		const int nBurnin, const int nSample, const int thin, const bool calculateLikelihood,
		const std::vector<double> &x, const bool usePrecision, const bool useMedian
	)
	: Model(nBurnin, nSample, thin, calculateLikelihood, useMedian), x_(x), usePrecision_(usePrecision)
	{
		step_.push_back(new SliceStep(new mean(this)));  // mean.
		step_.push_back(new SliceStep(new variance(this)));  // variance.
		step_.push_back(new FunctionStep(new precision(this)));  // precision.

		// set lower bound on variance.
		step_[1]->SetBounds(Util::dbl_min, Util::dbl_max);
	}

public:
	double Mean() const
	{
		return step_[0]->Value();
	}

	double Var() const
	{
		return step_[1]->Value();
	}

	double X(const int index) const
	{
		return x_[index];
	}

	double llike(const double mu, const double sd) const
	{
		const unsigned nElem = x_.size();
		double llike = 0.0;
		for (unsigned i = 0; i < nElem; ++i)
		{
			llike += Density::dnorm(x_[i], mu, sd, true);
		}
		return llike;
	}

	// log likelihood.
	/*virtual*/ double Llike(const SampleVector &p) const
	{
		double var = 0.0;
		if (usePrecision_)
		{
			var = 1.0 / boost::any_cast<double>(p[2]);  // p[2] = precision.
		}
		else
		{
			var = boost::any_cast<double>(p[1]);  // p[1] = variance.
		}
		const double sd = std::sqrt(var);

		const double mu = boost::any_cast<double>(p[0]);  // p[0] = mean.

		return llike(mu, sd);
	}

private:
	const std::vector<double> &x_;
	const bool usePrecision_;
};

double mean::llike(const double mu) const
{
	const double sd = std::sqrt(norm_->Var());
	return norm_->llike(mu, sd);
}

double mean::lPrior(const double mu0) const
{
	return Density::dnorm(mu0, 0.0, std::sqrt(1000.0), true);
}

double variance::llike(const double var) const
{
	const double mu = norm_->Mean();
	const double sd = std::sqrt(var);
	return norm_->llike(mu, sd);
}

double variance::lPrior(const double var0) const
{
	return Density::dnorm(var0, 0, 10, true);
}

const double precision::Function(const bool doCalc) const
{
	return 1.0 / norm_->Var();
}

}  // namespace local
}  // unnamed namespace

namespace my_mcmcpp {

// [ref] ${MCMC++_HOME}/examples/normal.cpp.
void normal_example()
{
	// mu = ?, sd = ?, n = 10.
	//const std::string dataFileName("./data/statistics/mcmcpp/normal_1.txt");
	// mu = 0, sd = 1, n = 20.
	const std::string dataFileName("./data/statistics/mcmcpp/normal_2.txt");

	DataTable<double> data(false, false);
	data.Read(dataFileName);
	const std::vector<double> x = data.ColumnVector(0);

	//
	const int nBurnin = 1000;
	const int nSample = 10000;
	const int thin = 5;
	const bool calculateLikelihood = true;

	const bool usePrecision = true;  // precision or variance.
	const bool useMedian = false;  // median or mean.

	local::NormalModel model(nBurnin, nSample, thin, calculateLikelihood, x, usePrecision, useMedian);
	std::cout << "start simulation ..." << std::endl;
	model.Simulation(std::cerr, true);
	std::cout << "end simulation ..." << std::endl;

	//
	std::cout << "DIC based on " << (usePrecision ? "precision" : "variance") << " and " << (useMedian ? "median" : "mean") << std::endl;
	model.Report(std::cout);
}

}  // namespace my_mcmcpp
