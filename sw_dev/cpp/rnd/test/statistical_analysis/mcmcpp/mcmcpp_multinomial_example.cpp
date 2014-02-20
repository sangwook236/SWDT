//#include "stdafx.h"
// MCMC++ includes.
#include <mcmc++/DataTable.h>
#include <mcmc++/Density.h>
#include <mcmc++/MCMC.h>
#include <mcmc++/intervals.h>
#include <mcmc++/statistics.h>
#include <mcmc++/util.h>
// Boost includes.
#include <boost/format.hpp>
// standard includes.
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>


namespace {
namespace local {

class MultinomialModel;

class lambda : public Parameter
{
public:
	lambda(MultinomialModel *multi, const int index)
	: Parameter("lambda"), multi_(multi), idx_(index)
	{
		std::ostringstream ost;
		ost << "lambda[" << idx_ + 1 << "]";
		SetLabel(ost.str());
		Assign(1);
	}

public:
	// likelihood associated with current parameter.
	/*virtual*/ double llike(const double lambda) const;
	// prior associated with current parameter.
	/*virtual*/ double lPrior(const double lambda0) const;

private:
	MultinomialModel *multi_;
	const int idx_;
};

class p : public Parameter
{
public:
	p(MultinomialModel *multi, const int index)
	: Parameter("p"), multi_(multi), idx_(index)
	{
		std::ostringstream ost;
		ost << "p[" << idx_ + 1 << "]";
		SetLabel(ost.str());
	}

public:
	// function used to update a deterministic node.
	/*virtual*/ const double Function(const bool doCalc = true) const;

private:
	MultinomialModel *multi_;
	const int idx_;
};

class MultinomialModel : public Model
{
public:
	MultinomialModel(const int nBurnin, const int nSample, const int thin, const std::vector<int> &x)
	: Model(nBurnin, nSample, thin, false, false), n_(x), size_(n_.size())
	{
		for (int i = 0; i < size_; ++i)
		{
			step_.push_back(new SliceStep(new lambda(this, i)));
			step_[i]->SetBounds(Util::dbl_min, Util::dbl_max);
		}
		for (int i = 0; i < size_; ++i)
		{
			step_.push_back(new FunctionStep(new p(this, i)));
		}
	}

public:
	int N(const int idx) const
	{
		return n_[idx];
	}

	double Lambda(const int idx) const
	{
		return step_[idx]->Value();
	}

	int Size(void) const
	{
		return size_;
	}

	double CumLambda(void) const
	{
		double sum = 0.0;
		for (int i = 0; i < size_; ++i)
		{
			sum += Lambda(i);
		}
		return sum;
	}

	void Report(std::ostream &outf)
	{
		std::vector<double> a(size_);
		double a0 = 0.0;
		for (int i = 0; i < size_; ++i)
		{
			a[i] = n_[i] + 1;
			a0 += n_[i] + 1;
		}
		std::vector<double> mu(size_);
		std::vector<double> sd(size_);
		for (int i = 0; i < size_; ++i)
		{
			mu[i] = a[i] / a0;
			sd[i] = std::sqrt(a[i] * (a0 - a[i]) / (Util::sqr(a0) * (a0 + 1)));
		}
		outf.precision(6);
		outf << std::endl << "Predicted: "
			<< std::endl << "  Mean:    " << mu
			<< std::endl << "  s.d.:    " << sd << std::endl;

		Model::Report(outf);
	}

	/*virtual*/ void Summarize(int i, std::ostream &outf)
	{
		const int n = nSample_ / nThin_;
		std::vector<double> x(n);
		for (int k = 0; k < n; ++k)
		{
			x[k] = boost::any_cast<double>(results_[k][i]);
		}

		SimpleStatistic xStat(x);
		// add size_ because of offset for p[].
		boost::format formatter("%|12| %|12| %|12| %|12| %|12| %|12|");
		outf << formatter % Label(i + size_) % xStat.Mean() % xStat.StdDev() % quantile(x, 0.025) % quantile(x, 0.5) % quantile(x, 0.975) << std::endl;
	}

protected:
	/*virtual*/ SampleVector Parameters(void) const
	{
		SampleVector x;
		// only interested in p, not lambda.
		ModelSteps::const_iterator i = step_.begin() + size_;
		ModelSteps::const_iterator iEnd = step_.end();
		for (; i != iEnd; ++i)
		{
			x.push_back((*i)->Value());
		}
		return x;
	}

private:
	const std::vector<int> n_;
	const int size_;
};

double lambda::llike(const double lambda) const
{
	return Density::dpois(multi_->N(idx_), lambda, true);
}

double lambda::lPrior(const double lambda0) const
{
	return Density::dgamma(lambda0, 1.0, 1.0, true);
}

const double p::Function(const bool doCalc) const
{
	const double k = multi_->Lambda(idx_);
	const double n = multi_->CumLambda();
	return k / n;
}

}  // namespace local
}  // unnamed namespace

namespace my_mcmcpp {

// [ref] ${MCMC++_HOME}/examples/multinomial.cpp.
void multinomial_example()
{
	const std::string fileName("./data/statistical_analysis/mcmcpp/multinomial.txt");;
	DataTable<int> data(false, false);
	data.Read(fileName);
	const std::vector<int> x = data.ColumnVector(0);

	//
	const int nBurnin = 1000;
	const int nSample = 10000;
	const int thin = 5;

	local::MultinomialModel model(nBurnin, nSample, thin, x);
	std::cout << "start simulation ..." << std::endl;
	model.Simulation(std::cerr, true);
	std::cout << "end simulation ..." << std::endl;

	//
	model.Report(std::cout);
}

}  // namespace my_mcmcpp
