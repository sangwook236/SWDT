//#include "stdafx.h"
// MCMC++ includes.
#include <mcmc++/DataTable.h>
#include <mcmc++/Density.h>
#include <mcmc++/MCMC.h>
#include <mcmc++/lot.h>
#include <mcmc++/statistics.h>
#include <mcmc++/util.h>
// standard includes.
#include <algorithm>
#include <iostream>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>


// argCheck_ controls whether row and column indexes are bounds checked before use.
// Defaults to 1 (true) unless NDEBUG is defined.
#if defined(NDEBUG)
#define argCheck_ 0
#else
#define argCheck_ 1
#endif

namespace {
namespace local {

lot rng_;
class MixtureAssert {};
class BadK {};

class MixtureModel;

class MuStep : public SliceStep
{
public:
	MuStep(Parameter *par, MixtureModel *mix, const int idx)
	: SliceStep(par), mix_(mix), idx_(idx)
	{}

public:
	// select new parameter from sampler.
	/*virtual*/ void DoStep(void);

private:
	MixtureModel *mix_;
	const int idx_;
};

class mean : public Parameter
{
public:
	mean(MixtureModel *mix, const int idx, const int k, const double min, const double max)
	: Parameter("mean"), mix_(mix), idx_(idx)
	{
		Util::Assert<MixtureAssert>(max > min);

		// fraction of distance from min to max.
		const double p = static_cast<double>(idx) / (k - 1);
		const double x = min + p * (max - min);
		Assign(x);

		std::ostringstream ost;
		ost << "mu[" << idx_ + 1 << "]";
		SetLabel(ost.str());
	}

public:
	// likelihood associated with current parameter.
	/*virtual*/ double llike(const double mu) const;
	// prior associated with current parameter.
	/*virtual*/ double lPrior(const double mu0) const;

private:
	MixtureModel *mix_;
	const int idx_;
};

class variance : public Parameter
{
public:
	variance(MixtureModel *mix, const int idx, const double variance) 
	: Parameter("variance"), mix_(mix), idx_(idx), sd_(10)
	{
		Assign(variance);

		std::ostringstream ost;
		ost << "sigma^2[" << idx_ + 1 << "]";
		SetLabel(ost.str());
	}

public:
	// likelihood associated with current parameter.
	/*virtual*/ double llike(const double var) const;
	// prior associated with current parameter.
	/*virtual*/ double lPrior(const double var0) const;

private:
	MixtureModel *mix_;
	const int idx_;
	const double sd_;
};

class lambda : public Parameter
{
public:
	lambda(MixtureModel *mix, const int index)
	: Parameter("lambda"), mix_(mix), idx_(index)
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
	MixtureModel *mix_;
	const int idx_;
};

class p : public Parameter
{
public:
	p(MixtureModel *mix, const int idx) 
	: Parameter("p"), mix_(mix), idx_(idx)
	{
		std::ostringstream ost;
		ost << "p[" << idx_ + 1 << "]";
		SetLabel(ost.str());
	}

public:
	// function used to update a deterministic node.
	/*virtual*/ const double Function(const bool doCalc = true) const;

private:
	MixtureModel *mix_;
	const int idx_;
};

class component : public ParameterT<int>
{
public:
	component(MixtureModel *mix, const int idx, const int nComp, const std::vector<double> &p) 
	: ParameterT<int>("component"), mix_(mix), idx_(idx)
	{
		const double u = rng_.uniform();
		double sum = p[0];
		int k = 0;
		while (sum < u)
		{
			++k;
			sum += p[k];
		}

		Assign(k);
	}

public:
	// likelihood associated with current parameter.
	/*virtual*/ double llike(const int k) const;
	// prior associated with current parameter.
	/*virtual*/ double lPrior(const int k0) const;
	// propose a new value for a parameter in an M-H step.
	/*virtual*/ int propose(int k) const;

private:
	MixtureModel *mix_;
	const int idx_;
};

class MixtureModel : public Model
{
public:
	MixtureModel(const int nBurnin, const int nSample, const int thin,
		const std::vector<double> &x, const int nComponents,
		const double minMu, const double maxMu, const double var, const std::string &logFileName
	)
	: Model(nBurnin, nSample, thin), x_(x), nComp_(nComponents), nElem_(x_.size()), logFile_(logFileName)
	{
		// means.
		for (int i = 0; i < nComp_; ++i)
		{
			step_.push_back(new MuStep(new mean(this, i, nComp_, minMu, maxMu), this, i));
		}
		// variances.
		varOffset_ = nComp_;
		Util::Assert<MixtureAssert>(!argCheck_ || (varOffset_ == step_.size()));
		for (int i = 0; i < nComp_; ++i)
		{
			// if all components are in equal frequency and have equal variance, total variance = n * (component variance).
			step_.push_back(new SliceStep(new variance(this, i, var / nComp_)));
			step_[varOffset_ + i]->SetBounds(Util::dbl_min, Util::dbl_max);
		}
		// mixture proportion.
		pOffset_ = varOffset_ + nComp_;
		Util::Assert<MixtureAssert>(!argCheck_ || (pOffset_ == step_.size()));
		for (int i = 0; i < nComp_; ++i)
		{
			step_.push_back(new FunctionStep(new p(this, i)));
		}
		// Poisson surrogates for p.
		lambdaOffset_ = pOffset_ + nComp_;
		Util::Assert<MixtureAssert>(!argCheck_ || (lambdaOffset_ == step_.size()));
		for (int i = 0; i < nComp_; ++i)
		{
			step_.push_back(new SliceStep(new lambda(this, i)));
			step_[lambdaOffset_ + i]->SetBounds(Util::dbl_min, Util::dbl_max);
		}
		// component indicators ==> for the number of mixture components (?).
		compOffset_ = lambdaOffset_ + nComp_;
		Util::Assert<MixtureAssert>(!argCheck_ || (compOffset_ == step_.size()));
		for (int i = 0; i < nElem_; ++i)
		{
			std::vector<double> p(nComp_);
			double sum = 0.0;
			for (int k = 0; k < nComp_; ++k)
			{
				p[k] = Density::dnorm(x_[i], Mu(k), Sd(k), false);
				sum += p[k];
			}
			for (int k = 0; k < nComp_; ++k)
			{
				p[k] /= sum;
			}
			step_.push_back(new MetroStepT<int>(new component(this, i, nComp_, p)));
		}
	}

public:
	int K(void) const
	{
		return nComp_;
	}

	int NElem(void) const
	{
		return nElem_;
	}

	int Idx(const int idx) const
	{
		return step_[compOffset_ + idx]->iValue();
	}

	double Lambda(const int k) const
	{
		return step_[lambdaOffset_ + k]->Value();
	}

	double CumLambda(void) const
	{
		double sum = 0.0;
		for (int k = 0; k < nComp_; ++k)
		{
			sum += Lambda(k);
		}
		return sum;
	}

	double P(const int k) const
	{
		return Lambda(k) / CumLambda();
	}

	double Mu(const int idx) const
	{
		return step_[idx]->Value();
	}

	double Sd(const int idx) const
	{
		return std::sqrt(Var(idx));
	}

	double Var(const int idx) const
	{
		return step_[varOffset_ + idx]->Value();
	}

	double X(const int idx) const
	{
		return x_[idx];
	}

	/*virtual*/ void Record(const SampleVector &p)
	{
		results_.push_back(p);
		write(p);
	}

protected:
	/*virtual*/ SampleVector Parameters(void) const
	{
		SampleVector x;
		for (int k = 0; k < nComp_; ++k)
		{
			x.push_back(Mu(k));
		}
		for (int k = 0; k < nComp_; ++k)
		{
			x.push_back(Var(k));
		}
		for (int k = 0; k < nComp_; ++k)
		{
			x.push_back(P(k));
		}
		return x;
	}

private:
	// since Parameters() above returns only the parameters we're interested,
	// we can simply iterate through the entire SampleVector
	void write(const SampleVector &p)
	{
		std::ofstream outf(logFile_.c_str(), std::ios::app);
		SampleIter i = p.begin();
		SampleIter iEnd = p.end();
		outf << std::endl;
		for (; i != iEnd; ++i)
		{
			outf << boost::any_cast<double>(*i) << " ";
		}
		outf.close();
	}

private:
	const std::vector<double> x_;
	const int nComp_;
	const int nElem_;
	const std::string logFile_;

	int varOffset_;
	int pOffset_;
	int compOffset_;
	int lambdaOffset_;
};

void MuStep::DoStep(void)
{
	if (0 == idx_)
	{
		const double muHi = mix_->Mu(1);
		SetBounds(Util::dbl_min, muHi);
	}
	else
	{
		const double muLo = std::max(mix_->Mu(idx_ - 1), Util::dbl_min);
		SetBounds(muLo, Util::dbl_max);
	}

	SliceStep::DoStep();
}

double mean::llike(const double mu) const
{
	const int nElem = mix_->NElem();
	double llike = 0.0;
	for (int i = 0; i < nElem; ++i)
	{
		const int idx = mix_->Idx(i);
		// only components whose index matches the mean index being updated contribute to the likelihood.
		if (idx_ == idx)
		{
			llike += Density::dnorm(mix_->X(i), mu, mix_->Sd(idx), true);
		}
	}
	return llike;
}

double mean::lPrior(const double mu0) const
{
	double lprior;
	if (0 == idx_)
	{
		lprior = (mu0 > mix_->Mu(1)) ? Util::log_dbl_min : Density::dnorm(mu0, 0.0, std::sqrt(1000.0), true);
	}
	else
	{
		lprior = (mu0 < mix_->Mu(idx_ - 1)) ? Util::log_dbl_min : Density::dnorm(mu0, mix_->Mu(idx_ - 1), std::sqrt(1000.0), true);
	}
	return lprior;
}

double variance::llike(const double var) const
{
	const int nElem = mix_->NElem();
	double llike = 0.0;
	for (int i = 0; i < nElem; ++i)
	{
		const int idx = mix_->Idx(i);
		// only components whose index matches the variance index being updated contribute to the likelihood.
		if (idx == idx_)
		{
			llike += Density::dnorm(mix_->X(i), mix_->Mu(idx), std::sqrt(var), true);
		}
	}
	return llike;
}

double variance::lPrior(const double var0) const
{
	return Density::dt(var0 / sd_, 1, true);
}

double lambda::llike(const double lambda) const
{
	const int nElem = mix_->NElem();
	int ct = 0;
	for (int i = 0; i < nElem; ++i)
	{
		const int idx = mix_->Idx(i);
		ct += (idx == idx_) ? 1 : 0;
	}
	return Density::dpois(ct, lambda, true);
}

double lambda::lPrior(const double lambda0) const
{
	return Density::dgamma(lambda0, 1.0, 1.0, true);
}

const double p::Function(const bool doCalc) const
{
	const double k = mix_->Lambda(idx_);
	const double n = mix_->CumLambda();
	return k / n;
}

double component::llike(const int k) const
{
	return Density::dnorm(mix_->X(idx_), mix_->Mu(k), mix_->Sd(k), true);
}

double component::lPrior(const int k0) const
{
	return std::log(mix_->P(k0));
}

int component::propose(int k) const
{
	const int newK = static_cast<int>(std::floor(rng_.uniform() * mix_->K()));
	Util::Assert<MixtureAssert>(!argCheck_ || ((newK >= 0) && (newK < mix_->K())));
	return newK;
}

}  // namespace local
}  // unnamed namespace

namespace my_mcmcpp {

// [ref] ${MCMC++_HOME}/examples/normal-mixture.cpp.
void normal_mixture_example()
{
	// component #1: mu = 0, sd = 1, n = 20.
	// component #2: mu = -2, sd = 0.8, n = 10.
	// component #3: mu = -3, sd = 1.5, n = 15.
	const std::string dataFileName("./data/statistical_analysis/mcmcpp/normal_mixture.txt");  // TODO [check] >> is a data file correct?
	const std::string logFileName("./data/statistical_analysis/mcmcpp/normal_mixture_log.txt");
	const int nComponents = 3;

	DataTable<double> data(false, false);
	data.Read(dataFileName);
	std::vector<double> x = data.ColumnVector(0);

	//
	{
		// make sure log file is empty before starting.
		std::ofstream outf(logFileName.c_str(), std::ios::trunc);
		for (int i = 0; i < nComponents; ++i)
		{
			outf << "mu[" << i+1 << "] ";
		}
		for (int i = 0; i < nComponents; ++i)
		{
			outf << "var[" << i+1 << "] ";
		}
		for (int i = 0; i < nComponents; ++i)
		{
			outf << "p[" << i+1 << "] ";
		}
		outf.close();
	}

	//
	const int nBurnin = 10;
	const int nSample = 100;
	const int thin = 1;

	const double minMu = Util::vectorMin(x);
	const double maxMu = Util::vectorMax(x);
	const SimpleStatistic xStat(x);

	local::MixtureModel model(nBurnin, nSample, thin, x, nComponents, minMu, maxMu, xStat.Variance(), logFileName);
	std::cout << "start simulation ..." << std::endl;
	model.Simulation(std::cerr, true);
	std::cout << "end simulation ..." << std::endl;

	//
	model.Report(std::cout);
	Util::Assert<local::MixtureAssert>(!argCheck_ || false);
}

}  // namespace my_mcmcpp
