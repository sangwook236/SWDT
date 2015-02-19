/// \file   MCMC.cpp
/// \brief  Definitions of classes for MCMC evaluation of Bayesian models
///
/// The Parameter, Step, and Model classes defined here are the core classes
/// that do all of the work. To implement a Bayesian model using these classes
/// each parameter in the model should be derived from Paramater and should
/// override llike(), at a minimum. lprior() should be overridden for
/// anything other than a flat prior. (Notice that the prior will be
/// improper unless overriden when the parameter has an unbounded domain.)
/// The model is derived from Model, and each parameter is pushed onto a
/// stack (step_), with a specified Step type (MetroStep, SliceStep, or
/// FunctionStep).
///
/// \author Kent Holsinger
/// \date   2005-05-18
///

// This file is part of MCMC++, a library for constructing C++ programs
// that implement MCMC analyses of Bayesian statistical models.
// Copyright (c) 2004-2006 Kent E. Holsinger
//
// MCMC++ is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// MCMC++ is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with MCMC++; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//

// standard includes
#include <iostream>
// local includes
#include "mcmc++/Density.h"
#include "mcmc++/MCMC.h"
#include "mcmc++/intervals.h"

using std::copy;
using std::endl;
using std::ostream;
using std::vector;
using boost::any;
using boost::any_cast;
using Density::dbeta;
using Density::dnorm;

/// argCheck_ controls whether arguments are bounds checked
/// before use
///
/// Defaults to 1 (true) unles NDEBUG is defined
///
#if defined(NDEBUG)
#define argCheck_ 0
#else
#define argCheck_ 1
#endif

struct BadValue {};
struct BadCT {};

/// Macro to determine whether a value isnan() and print __FILE__ and 
/// __LINE__ if it is (disabled when argCheck == 0 (i.e., when NDEBUG
/// is defined during compilation
///
#if defined(HAVE_ISNAN) && !defined(MINGW32) && !defined(__CYGWIN__)
extern "C" { int isnan(double x); }
#define checkValue_(x) \
  if (argCheck_ && isnan(x)) {       \
    std::cout << __FILE__ << ": " << __LINE__ << std::endl;  \
  } \
  Util::Assert<BadValue>(!argCheck_ || !isnan(x))
#else
#define checkValue_(x) ;
#endif

// MCMC uses the Mersenne twister on (0,1) by default, because the slice 
// sampler requires it. Notice that the non-uniform RNGs in lot are 
// validated with the Mersenne twister on [0,1). It's conceivable that
// this might affect the accuracy of Metropolis-Hastings steps, but I 
// don't <b>think</b> it will make much difference. The underlying sequence 
// of integers is the same. Each value from the (0,1) version of the MT 
// exceeds the corresponding value from the [0,1) version by 
// \f$0.5/4294967296.0 < 1.2 \times 10^{-10}\f$. If you're paranoid,
// you can override the proposal functions in Parameter and Set_MT(ZERO).
//
namespace {

  lot rng_(lot::RAN_MT, lot::OPEN);   ///< RNG used in all classes (static)

}

/// default "zero" value for safeFreq()
///
const double MCMC_ZERO_FREQ = 1.00e-14;

/// Returns a reference to the internal random number generator
///
lot& GetRNG(void) {
  return rng_;
}

/// Ensures zero < p < 1-zero
///
/// \param p    Frequency to guard
/// \param zero Minimum value of p allowed
///
/// Assumes all elements of p are in [0, 1]
///
double safeFreq(const double p, const double zero) {
  return p*(1.0 - 2.0*zero) + zero;
}

/// Converts a safeFreq()ed x back
///
/// \param x    Frequency to convert back
/// \param zero Value used in original guard (not verified)
///
double invSafeFreq(const double x, const double zero) {
  double p = (x - zero) / (1.0 - 2.0 * zero);
  return p;
}

/// Ensures that all values in vector are greater than zero and
/// less than 1.0 - n*zero, where n is p.size()
///
/// \param p    Frequency vector to guard
/// \param zero Value used for guard (MCMC_ZERO_FREQ default)
///
/// Assumes all elements of p are in [0, 1]
///
std::vector<double>
safeFreq(const std::vector<double>& p, const double zero) {
  std::vector<double> q(p.size());
  int nElem = q.size();
  for (unsigned i = 0; i < nElem; i++) {
    q[i] = p[i] * (1.0 - nElem*zero) + zero;
  }
  return q;
}

/// Inverts a safeFreq()ed vector
///
/// \param x        The vector to invert
/// \param zero     The original guard (not checked)
///
std::vector<double>
invSafeFreq(const std::vector<double>& x,
            const double zero) 
{
  unsigned nElem = x.size();
  std::vector<double> q(nElem);
  for (unsigned k = 0; k < nElem; ++k) {
    q[k] = (x[k] - zero)/(1.0 - nElem*zero);
  }
  return q;
}


/// Propose a new beta variate for an M-H step
///
/// \param mean      Mean for beta distribution
/// \param denom     Effective sample size
/// \param tolerance Minimum value allowed (to avoid exact 0s and 1s)
///
/// The parameters of the beta distribution are given by
///
/// \f[\alpha = denom \times mean\f]
/// \f[\beta  = denom \times (1-mean)\f]
///
/// where alpha and beta are guarded by safeBetaPar()
///
double proposeBeta(const double mean, const double denom, 
                       const double tolerance) 
{
  double alphaProposal, betaProposal;
  alphaProposal = safeBetaPar(denom * mean);
  betaProposal = safeBetaPar(denom - alphaProposal);
  // select a new value for p
  double p = rng_.beta(alphaProposal, betaProposal);
  return safeFreq(p, tolerance);
}

/// lQ for a beta proposal
///
/// \param newMean   New value proposed
/// \param oldMean   Old value from which proposed
/// \param denom     Effective sample size
/// \param tolerance Minimum value allowed in original proposal
///
double logQBeta(const double newMean, const double oldMean,
                    const double denom, const double tolerance)
{
  double alphaProposal = safeBetaPar(denom * oldMean);
  double betaProposal = safeBetaPar(denom - alphaProposal);
  double p = invSafeFreq(newMean, tolerance);
  double value = dbeta(p, alphaProposal, betaProposal, true);
  return Util::safeLog(value);
}

/// Propose a new Dirichlet vector for an M-H step
///
/// \param q         Mean of the proposal distribution
/// \param denom     Effective sample size
/// \param tolerance Minimum value allowed (to avoid exact 0s and 1s)
///
/// The parameters of the beta distribution are given by
///
/// \f[\alpha = denom \times mean\f]
/// \f[\beta  = denom \times (1-mean)\f]
///
/// where alpha and beta are guarded by safeBetaPar()
///
std::vector<double>
proposeDirch(const std::vector<double>& q, const double denom,
             const double tolerance) 
{
  unsigned nElem = q.size();
  std::vector<double> alpha(nElem);
  for (unsigned i = 0; i < nElem; i++) {
    alpha[i] = q[i]*denom;
  }
  safeDirchPar(alpha);
  std::vector<double> p = rng_.dirichlet(alpha);
  std::vector<double> r = safeFreq(p, tolerance);
  return r;
}

/// lQ for a Dirichlet proposal
///
/// \param newMean   New value proposed
/// \param oldMean   Old value from which proposed
/// \param denom     Effective sample size
/// \param tolerance Minimum value allowed in original proposal
///
double
logQDirch(const std::vector<double>& newMean,
          const std::vector<double>& oldMean,
          const double denom, const double tolerance)
{
  int nElem = newMean.size();
  std::vector<double> alpha(nElem);
  for (unsigned k = 0; k < nElem; ++k) {
    alpha[k] = denom * oldMean[k];
  }
  std::vector<double> p = invSafeFreq(newMean, tolerance);
  double value = Density::ddirch(p, alpha, true);
  return Util::safeLog(value);
}

/// Propose a new normal variate for an M-H step
///
/// \param mean     Mean of the proposal distribution
/// \param variance Variance of the proposal distribution
///
double proposeNorm(const double mean, const double variance) {
  return rng_.norm(mean, variance);
}

/// lQ for a normal proposal
///
/// \param newMean   New value proposed
/// \param oldMean   Old value from which proposed
/// \param variance  Variance of the proposal distribution
///
double logQNorm(const double newMean, const double oldMean,
                    const double variance)
{
  double value = dnorm(newMean, oldMean, variance, true);
  return Util::safeLog(value);
}

/// Ensures MinBetaPar <= x <= MaxBetaPar
///
/// \param x    Frequency to guard
///
double safeBetaPar(const double x) {
  using MCMC::MinBetaPar;
  using MCMC::MaxBetaPar;
  return (x < MinBetaPar) ? MinBetaPar : (x > MaxBetaPar) ? MaxBetaPar : x;
}

/// Ensures MinBetaPar <= x <= MaxBetaPar for all elements of a 
/// Dirichlet parameter vector
///
/// \param x    Frequency to guard
///
void safeDirchPar(std::vector<double>& x) {
  using MCMC::MinDirchPar;
  using MCMC::MaxDirchPar;
  int nElem = x.size();
  for (int i = 0; i < nElem; ++i) {
    x[i] = (x[i] < MinDirchPar) ? MinDirchPar : 
      (x[i] > MaxDirchPar) ? MaxDirchPar : x[i];
  }
}

/// Constructor
///
/// \param label    a string identifier for the parameter
///
/// Used internally to construct Parameter classes. Not intended for direct
/// use.
ParameterBase::ParameterBase(std::string label)
  : label_(label)
{}

/// Destructor
///
ParameterBase::~ParameterBase(void) 
{}

/// Adapt the proposal distribution in a Metropolis-Hastings sampler
///
/// \param accept  the current acceptance percentage
///
void 
ParameterBase::Adapt(const double accept) {}

/// internal typedef, defined here so that it is invisible and
/// inaccessible to users of the library
///
typedef ParameterT<double> dPar;
/// internal typedef, defined here so that it is invisible and
/// inaccessible to users of the library
///
typedef ParameterT<int> iPar;
/// internal typedef, defined here so that it is invisible and
/// inaccessible to users of the library
///
typedef ParameterT<std::vector<double> > dVecPar;
/// internal typedef, defined here so that it is invisible and
/// inaccessible to users of the library
///
typedef ParameterT<std::vector<int> > iVecPar;
/// internal typedef, defined here so that it is invisible and
/// inaccessible to users of the library
///
typedef ParameterT<boost::any> aPar;

/// sets value of the parameter
///
/// \param value the value to be assigned
///
void
ParameterBase::Assign(const boost::any& value) {
  struct UnsupportedParameterType {} ;

  if (dPar* q = dynamic_cast<dPar*>(this)) {
    return q->Assign(any_cast<double>(value));
  } else if (iPar* q = dynamic_cast<iPar*>(this)) {
    return q->Assign(any_cast<int>(value));
  } else if (dVecPar* q = dynamic_cast<dVecPar* >(this)) {
    return q->Assign(any_cast<vector<double> >(value));
  } else if (iVecPar* q = dynamic_cast<iVecPar* >(this)) {
    return q->Assign(any_cast<vector<int> >(value));
  } else if (aPar* q = dynamic_cast<aPar*>(this) ) {
    return q->Assign(value);
  } else {
    throw UnsupportedParameterType();
  }
}


/// Returns string identifying current parameter
///
/// The default value is an empty string. Replace it by including an
/// argument to the constructor 
///
std::string 
ParameterBase::Label(void) const {
  return label_;
}

/// Set the label associated with this parameter
///
/// \param label  the label string
///
void 
ParameterBase::SetLabel(const std::string& label) {
  label_ = label;
}

/// internal typedef, defined here so that it is invisible and
/// inaccessible to users of the library
///
typedef const ParameterT<double> cdPar;
/// internal typedef, defined here so that it is invisible and
/// inaccessible to users of the library
///
typedef const ParameterT<int> ciPar;
/// internal typedef, defined here so that it is invisible and
/// inaccessible to users of the library
///
typedef const ParameterT<std::vector<double> > cdVecPar;
/// internal typedef, defined here so that it is invisible and
/// inaccessible to users of the library
///
typedef const ParameterT<std::vector<int> > ciVecPar;
/// internal typedef, defined here so that it is invisible and
/// inaccessible to users of the library
///
typedef const ParameterT<boost::any> caPar;

/// Value of the parameter
///
const boost::any 
ParameterBase::Value(void) const {
  struct UnsupportedParameterType {} ;

  if (cdPar* q = dynamic_cast<cdPar*>(this)) {
    return q->Value();
  } else if (ciPar* q = dynamic_cast<ciPar*>(this)) {
    return q->Value();
  } else if (cdVecPar* q = dynamic_cast<cdVecPar* >(this)) {
    return q->Value();
  } else if (ciVecPar* q = dynamic_cast<ciVecPar* >(this)) {
    return q->Value();
  } else if (caPar* q = dynamic_cast<caPar*>(this) ) {
    return q->Value();
  } else {
    throw UnsupportedParameterType();
  }
}

/// Constructor
///
/// Used internally to construct Step classes. Not intended for direct
/// use.
///
StepBase::StepBase(ParameterBase* parameter, unsigned long accept, 
                   unsigned long ct, const double w, const double m, 
                   const double lowBound, const double highBound) 
  : par_(parameter), accept_(accept), ct_(ct), w_(w), m_(m),
    lowBound_(lowBound), highBound_(highBound)
{}

/// Destructor
///
StepBase::~StepBase(void) 
{}

/// Number of proposals accepted (for M-H)
///
int 
StepBase::accept(void) const {
  return accept_;
}

/// Label of parameter associated with this step
///
std::string 
StepBase::Label(void) const {
  return par_->Label();
}

/// Parameter associated with this step
///
ParameterBase*
StepBase::Par(void) const {
  return par_;
}

/// Reset acceptance statistics (for M-H)
///
void 
StepBase::ResetAccept(void) {
  accept_ = 0;
}

/// Set the label associated with this step
///
/// \param label  the label string
///
void 
StepBase::SetLabel(const std::string& label) {
  par_->SetLabel(label);
}

/// Returns parameter value of the current step
///
/// Assumes value stored as boost::any
///
const boost::any 
StepBase::aValue(void) const {
  return par_->Value();
}

/// Returns parameter value of the current step
///
/// Assumes value stored as double
///
const double
StepBase::Value(void) const {
  dPar* q = dynamic_cast<dPar*>(par_);
  return q->Value();
}

/// Returns parameter value of the current step
///
/// Assumes value stored as integer
///
const int 
StepBase::iValue(void) const {
  iPar* q = dynamic_cast<iPar*>(par_);
  return q->Value();
}

/// Returns parameter value of the current step
///
/// Assumes value stored as vector<double>
///
const vector<double>
StepBase::dVecValue(void) const {
  dVecPar* q = dynamic_cast<dVecPar*>(par_);
  return q->Value();
}

/// Returns parameter value of the current step
///
/// Assumes value stored as vector<int>
///
const vector<int>
StepBase::iVecValue(void) const {
  iVecPar* q = dynamic_cast<iVecPar*>(par_);
  return q->Value();
}


const double SliceStep::DefaultW = 1.0;   ///< default "step out" width
const long SliceStep::MUnbounded = -1;    ///< unlimited number of steps out allowed by default

/// Constructor
///
/// \param parameter  Pointer to the parameter associated with this step
///
/// The "step out" width is set to 1, unless the parameter has defined
/// W() and returns a value > 0.0. The "step out" procedure is set to do
/// an unlimited number of steps, unless the parameter has defined M()
/// and returns a value > 0.
///
/// Notice that the defaults will result in the slice sampler adapting
/// the "step out" width as the simulation proceeds. Adapting the width is
/// permissible <b>only</b> when the distribution is unimodal. If you're not
/// sure that your distribution is unimodal, you should define W() and M() to
/// return values that seem reasonable to you.
///
SliceStep::SliceStep(Parameter* parameter) 
  : Step<double>(parameter, SliceStep::DefaultW, SliceStep::MUnbounded, 
                 -Util::dbl_max, Util::dbl_max), dbl_(parameter),
    llike_(&Parameter::llike), lPrior_(&Parameter::lPrior)
{
  ct_ = 0;   // I don't think this should be necessary, since ct_(0) is
             // included in the initialization for Step(), but it fails
             // on some calls if it's not included
}

/// Sets upper and lower limits on values allowed
///
/// \param low  Lower bound on parameter
/// \param high Upper bound on parameter
///
/// For parameters that are bounded, e.g., frequencies on [0,1], setting
/// bounds on allowable values helps avoid numerical problems in evaluating
/// likelihoods and priors.
///
void 
SliceStep::SetBounds(const double low, const double high) {
  lowBound_ = low;
  highBound_ = high;
  if ((lowBound_ > -Util::dbl_max) && (highBound_ < Util::dbl_max)) {
    w_ = (highBound_ - lowBound_)/10;
  }
}

/// Reset step width
///
/// \param w  New step width
///
void 
SliceStep::SetW(const double w) {
  w_ = w;
}

/// Reset number of steps out allowed
///
/// \param m  New number of steps out allowed
///
void 
SliceStep::SetM(const int m) {
  m_ = m;
}

/// Assign value to parameter associated with this step
///
/// \param x  The value to assign
///
void 
SliceStep::Assign(const double x) {
  dbl_->Assign(x);
}

/// Return a new value from the slice sampler
///
void
SliceStep::DoStep(void) {
  double x = dbl_->Value();
  SetSlice(x);
  dbl_->Assign(Sample(x));
}

/// "stepping out" version of slice sampler
///
/// \param x   current value of parameter
///
/// N.B.: uniform on (0,1), not [0,1) or [0,1], required throughout 
/// SetSlice() and Sample()
///
void
SliceStep::SetSlice(const double x0) {
  double u = rng_.uniform();
  checkValue_(u);
  checkValue_(l_);
  checkValue_(r_);
  checkValue_(x0);
  checkValue_(w_);
  l_ = std::max(x0 - w_*u, lowBound_);
  r_ = std::min(l_ + w_, highBound_);
  double v = rng_.uniform();
  checkValue_(v);
  if (m_ == SliceStep::MUnbounded) {
    z_ = (dbl_->*llike_)(x0) + (dbl_->*lPrior_)(x0) - rng_.exponential(1.0);
    checkValue_(z_);

    while ((l_ > lowBound_) && (z_ < (dbl_->*llike_)(l_) 
                                + (dbl_->*lPrior_)(l_))) 
      {
        l_ = std::max(l_ - w_, lowBound_);
        checkValue_(l_);
      }
    
    while ((r_ < highBound_) && (z_ < (dbl_->*llike_)(r_) 
                                 + (dbl_->*lPrior_)(r_))) 
      {
        r_ = std::min(r_ + w_, highBound_);
        checkValue_(r_);
      }

    // adjust width
    ++ct_;
    w_ = (w_*ct_ + (r_ - l_))/(ct_ + 1);
    checkValue_(ct_);
 
    checkValue_(w_);
  } else {
    int j = static_cast<int>(floor(m_*v));
    int k = static_cast<int>(m_ -  1) - j;
    z_ = (dbl_->*llike_)(x0) + (dbl_->*lPrior_)(x0) - rng_.exponential(1.0);
    checkValue_(z_);
    while ((l_ > lowBound_) 
           && (j > 0) && (z_ < (dbl_->*llike_)(l_) + (dbl_->*lPrior_)(l_)))
      {
        l_ = std::max(l_ - w_, lowBound_);
        checkValue_(l_);
        --j;
      }
  
    while ((r_ < highBound_)
           && (k > 0) && (z_ < (dbl_->*llike_)(r_) + (dbl_->*lPrior_)(r_))) 
      {
        r_ = std::min(r_ + w_, highBound_);
        checkValue_(r_);
        --k;
      }
  }
}

double
SliceStep::Sample(const double x0) {
  double u = rng_.uniform();
  double x1 = l_ + u*(r_ - l_);
  while (z_ > (dbl_->*llike_)(x1) + (dbl_->*lPrior_)(x1)) {
    if (x1 < x0) {
      l_ = x1;
    } else {
      r_ = x1;
    }
    u = rng_.uniform();
    x1 = l_ + u*(r_ - l_);
  }
  return x1;
}

/// Value of parameter associated with this step
///
const double
SliceStep::Value(void) const {
  return dbl_->Value();
}

namespace {
  boost::format formatter_("%|12| %|12| %|12| %|12| %|12| %|12|");
  boost::format dic_("%|12| %|12|");
}

// N.B.: nBurnin + nSample <= Util::ulong_max required (but probably
// ensured by use of vector<vector<double> > to hold parameter
// results)

/// Constructor
///
/// \param nBurnin  Number of iterations for "burn in"
/// \param nSample  Number of iterations for "sample"
/// \param nThin    Keep every nThin'th sample
/// \param calc     false == no lLike not overridden,
///                 true == lLike overriden (there's probably a way to
///                 figure this out, but I haven't worked on it yet).
///                 Default is false
/// \param useMedian  false == use posterior mean in DIC calculation,
///                   true == use posterior median in DIC calculation.
///                   Default is false.
///
Model::Model(const int nBurnin, const int nSample, const int nThin,
             const bool calc, const bool useMedian)
  : nBurnin_(nBurnin), nSample_(nSample), nThin_(nThin), nElem_(0),
    summaryFormat_(&formatter_), pd_(0), calculateLikelihood_(calc),
    useMedian_(useMedian)
{}

/// Destructor
///
Model::~Model(void) {}

/// Invoke simulation to do the analysis
///
/// \param outf  Output stream for progress display
/// \param interimReport  Produce progress display?
///
void 
Model::Simulation(std::ostream& outf, const bool interimReport) {
  nElem_ = step_.size();
  for (int i = 0; i < nBurnin_; ++i) {
    Generation();
    if (interimReport) {
      InterimReport(outf, "Burnin...", i, nBurnin_);
    }
  }
  Reset();
  for (int i = 0; i < nSample_; ++i) {
    Generation();
    if (interimReport) {
      InterimReport(outf, "Sample...", i, nSample_);
    }
    if (((i+1) % nThin_) == 0) {
      Record(Parameters());
    }
  }
}

/// Keep track of current parameter values
///
/// \param p   Current parameter values
///
/// If you're analysing a big model with lots of parameters, only some
/// of which are of primary interest, you may want to override Record() to
/// either keep track only of the few parameters that are of primary 
/// interest, either discarding the others or writing them to disk.
///
void
Model::Record(const SampleVector& p) {
  results_.push_back(p);
  if (calculateLikelihood_) {
    likelihood_.Add(Llike(p));
  }
}

/// Produce a progress display
///
/// If you're writing a sampler under a GUI and want a progress display,
/// you'll definitely want to override this.
///
void
Model::InterimReport(std::ostream& outf, const std::string header, 
                     const int progress, const int goal) 
{
  if (!pd_) {
    outf << endl << header;
    pd_ = new boost::progress_display(goal, outf);
  }
  ++(*pd_);
  if (progress == goal-1) {
    delete pd_;
    pd_ = 0;
  }
}

/// Produce a final report
///
/// \param outf     The stream on which the report is to be written
/// \param lastPar  -1 = all parameters, k+1 = k parameters 
///
/// The report includes a header, the label for each parameter, and
/// the posterior mean, standard deviation, 2.5%, 50%, and 97.5%tiles.
///
void
Model::Report(std::ostream& outf, const int lastPar) {
  ReportHead(outf);
  unsigned nPars = (lastPar < 0) ? results_[0].size() : lastPar;
  for (unsigned i = 0; i < nPars; ++i) {
    Summarize(i, outf);
  }
  ReportDic(outf);
}

/// Set the number of iterations in the burn-in period
///
/// \param nBurnin   the number of burn-in iterations
///
void
Model::SetBurnIn(const int nBurnin) {
  nBurnin_ = nBurnin;
}

/// Set the number of iterations in the sample period
///
/// \param nSample   the number of sample iterations
///
void
Model::SetSample(const int nSample) {
  nSample_ = nSample;
}

/// Set the thinning interval (the interval between saved samples)
///
/// \parm thin   the thinning interval
///
void
Model::SetThin(const int thin) {
  nThin_ = thin;
}

/// Produce the report header
///
/// \param outf The stream on which the report is being written
/// 
void
Model::ReportHead(std::ostream& outf) {
  outf << "Posterior summary..." << endl
       << *summaryFormat_ 
    % "Parameter" % "Mean" % "s.d." % "2.5%" % "50%" % "97.5%" << endl;
  outf << *summaryFormat_
    % "---------" % "------------" % "------------" % "------------" 
    % "------------" % "------------" << endl;
}

/// Produce the summary statistics for each parameter
///
/// \param i  Index of the parameter being reported on
/// \param outf The stream on which the report is being written
///
void
Model::Summarize(const int i, std::ostream& outf) {
    int n = nSample_/nThin_;
    vector<double> x(n);
    for (int k = 0; k < n; ++k) {
      x[k] = any_cast<double>(results_[k][i]);
    }
    SimpleStatistic xStat(x);
    outf << *summaryFormat_ % Label(i)
      % xStat.Mean() % xStat.StdDev() % quantile(x, 0.025) 
      % quantile(x, 0.5) % quantile(x, 0.975) << endl;
}

/// Produce the parameter label
///
/// \param i  Index of the parameter
///
std::string 
Model::Label(const int i) const {
  return step_[i]->Label();
}


/// Likelihood
///
/// \param par  A vector of parameters from which to calculate the likelihood
///
double
Model::Llike(const SampleVector& par) const {
  return 0.0;
}

/// Calculate and report DIC statistics
///
void
Model::ReportDic(std::ostream& outf) {
  if (!calculateLikelihood_) {
    return;
  }
  SampleIter sTer = results_[0].begin();
  SampleIter sEnd = results_[0].end();
  SampleVector pMean(nElem_);
  if (useMedian_) {   // use posterior median for Dhat
    int n = results_.size();
    for (int j = 0; j < nElem_; ++j) {
      vector<double> x(n);
      for (int i = 0; i < n; ++i) {
        x[i] = any_cast<double>(results_[i][j]);
      }
      pMean[j] = quantile(x, 0.5);
    }
  } else {   // use posterior mean for Dhat (default)
    vector<SimpleStatistic> stats(nElem_);
    ResultsIter iTer = results_.begin();
    ResultsIter iEnd = results_.end();
    for (; iTer != iEnd; ++iTer) {
      SampleVector par(*iTer);
      SampleIter pTer = par.begin();
      SampleIter pEnd = par.end();
      for (unsigned j = 0; j < nElem_; ++j) {
        stats[j].Add(any_cast<double>(par[j]));
      }
    }
    for (unsigned j = 0; j < nElem_; ++j) {
      pMean[j] = stats[j].Mean();
    }
  }
  double Dbar = -2.0*likelihood_.Mean();
  double Dhat = -2.0*Llike(pMean);
  double pD = Dbar - Dhat;
  double DIC = Dbar + pD;
  outf << dic_ % "---------" % "------------" << endl;
  outf << dic_ % "Dbar" % Dbar << endl;
  outf << dic_ % "Dhat" % Dhat << endl;
  if (0) {
    // deviance = -2*likelihood
    // var(deviance) = 4*var(likelihood)
    // pD2 = (1/2)var(deviance) -- Gelman et al., 2nd edition, p. 182
    double pD2 = 2.0*likelihood_.Variance();
    outf << dic_ % "pD(1)" % pD << endl;
    outf << dic_ % "pD(2)" % pD2 << endl;
  } else {
    outf << dic_ % "pD" % pD << endl;
  }
  outf << dic_ % "DIC" % DIC << endl;
}

/// percent -- integer/integer
///
/// \param k  Count
/// \param n  Sample size
///
double 
Model::percent(const int k, const int n) const {
  return 100.0*static_cast<double>(k)/n;
}

/// percent -- average percent across a vector
///
/// \param x  Vector of counts
/// \param n  Sample size
///
double 
Model::percent(const std::vector<int>& x, const int n) const {
  int nElem = x.size();
  SimpleStatistic pStat;
  for (int i = 0; i < nElem; ++i) {
    pStat.Add(percent(x[i], n));
  }
  return pStat.Mean();
}

/// percent -- average percent across a vector of vectors
///
/// \param x  Vector of vector of counts
/// \param n  Sample size
///
double 
Model::percent(const std::vector<std::vector<int> >& x, const int n) const {
  int nRows = x.size();
  SimpleStatistic pStat;
  for (int i = 0; i < nRows; ++i) {
    int nCols = x[i].size();
    for (int j = 0; j < nCols; ++j) {
      pStat.Add(percent(x[i][j], n));
    }
  }
  return pStat.Mean();
}

void
Model::Reset(void) {
  ModelSteps::const_iterator i = step_.begin();
  ModelSteps::const_iterator iEnd = step_.end();
  for (; i != iEnd; ++i) {
    (*i)->ResetAccept();
  }
}

void
Model::Generation(void) {
  ModelSteps::const_iterator i = step_.begin();
  ModelSteps::const_iterator iEnd = step_.end();
  for (; i != iEnd; ++i) {
    (*i)->DoStep();
  }
}

/// Return parameter vector associated with this step
///
/// Values in the sample vector are stored as boost::any. Retrieving them
/// will require an appropriate boost::any_cast<>
///
SampleVector
Model::Parameters(void) const {
  SampleVector x;
  ModelSteps::const_iterator i = step_.begin();
  ModelSteps::const_iterator iEnd = step_.end();
  for (; i != iEnd; ++i) {
    x.push_back((*i)->aValue());
  }
  return x;
}

