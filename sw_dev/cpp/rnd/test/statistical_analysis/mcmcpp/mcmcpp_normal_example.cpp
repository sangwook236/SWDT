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
	double llike(const double mu) const;
	double lPrior(const double mu0) const;

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
	double llike(const double var) const;
	double lPrior(const double var) const;

private:
	NormalModel *norm_;
};

class precision : public Parameter
{
public:
	precision(NormalModel *norm)
	: Parameter("precision"), norm_(norm)
	{}
	const double Function(const bool doCalc = true) const;

private:
	NormalModel *norm_;
};

class NormalModel : public Model
{
public:
	NormalModel(
		const int nBurnin, const int nSample, const int thin, 
		const std::vector<double> &x, const bool usePrecision, const bool useMedian
	)
	: Model(nBurnin, nSample, thin, true, useMedian), x_(x), usePrecision_(usePrecision)
	{
		step_.push_back(new SliceStep(new mean(this)));
		step_.push_back(new SliceStep(new variance(this)));
		step_.push_back(new FunctionStep(new precision(this)));
		// set lower bound on variance.
		step_[1]->SetBounds(Util::dbl_min, Util::dbl_max);
	}

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
		double llike = 0;
		unsigned nElem = x_.size();
		for (unsigned i = 0; i < nElem; ++i)
		{
			llike += Density::dnorm(x_[i], mu, sd, true);
		}
		return llike;
	}

	double Llike(const SampleVector &p) const
	{
		double a;
		if (usePrecision_)
		{
			a = 1 / boost::any_cast<double>(p[2]);  // p[2] == precision
		}
		else
		{
			a = boost::any_cast<double>(p[1]);  // p[1] == variance
		}
		double mu = boost::any_cast<double>(p[0]);
		double sd = std::sqrt(a);
		return llike(mu, sd);
	}

private:
	const std::vector<double> &x_;
	bool usePrecision_;

};

double mean::llike(const double mu) const
{
	const double sd = std::sqrt(norm_->Var());
	return norm_->llike(mu, sd);
}

double mean::lPrior(const double mu) const
{
	return Density::dnorm(mu, 0.0, std::sqrt(1000.0), true);
}

double variance::llike(const double var) const
{
	const double mu = norm_->Mean();
	const double sd = std::sqrt(var);
	return norm_->llike(mu, sd);
}

double variance::lPrior(const double var) const
{
	return Density::dnorm(var, 0, 10, true);
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
	const bool usePrecision = true;  // precision or variance.
	const bool useMedian = false;  // median or mean.

	DataTable<double> data(false, false);
	const std::string fileName("./data/statistical_analysis/mcmcpp/norm.txt");
	data.Read(fileName);
	std::vector<double> x = data.ColumnVector(0);

	const int nBurnin = 1000;
	const int nSample = 10000;
	const int thin = 5;

	local::NormalModel model(nBurnin, nSample, thin, x, usePrecision, useMedian);
	model.Simulation(std::cerr, true);
	std::cout << "DIC based on " << (usePrecision ? "precision" : "variance")
		<< " and " << (useMedian ? "median" : "mean") << std::endl;
	model.Report(std::cout);
}

}  // namespace my_mcmcpp
