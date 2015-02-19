///
/// \file   MCMC.h
/// \brief  Definitions of classes for MCMC evaluation of Bayesian models
///
/// The ParameterT, StepT, and Model classes defined here are the core classes
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
/// \date   2004-07-03
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

#if !defined(__MCMC_H)
#define __MCMC_H

// Boost includes
#include <boost/any.hpp>
#include <boost/format.hpp>
#include <boost/progress.hpp>
// local includes
#include "mcmc++/DataTable.h"
#include "mcmc++/lot.h"
#include "mcmc++/util.h"
#include "mcmc++/statistics.h"

extern const double MCMC_ZERO_FREQ;

namespace MCMC {
  const double MinBetaPar = 1.0e-1;   ///< minimum value allowed for beta parameter
  const double MaxBetaPar = 1.0e4;    ///< maximum value allowed for beta parameter
  const double MinDirchPar = 1.0e-1;  ///< minimum value allowed for Dirichlet parameter
  const double MaxDirchPar = 1.0e4;   ///< maximum value allowed for Dirichlet parameter
}

lot& GetRNG();
double safeFreq(double p, double zero = MCMC_ZERO_FREQ);
double invSafeFreq(double x, double zero = MCMC_ZERO_FREQ); 
std::vector<double> safeFreq(const std::vector<double>& p,
                             const double zero);
std::vector<double> invSafeFreq(const std::vector<double>& x,
                                const double zero = MCMC_ZERO_FREQ);

double proposeBeta(double mean, double denom, double tolerance);
double logQBeta(double newMean, double oldMean, double denom, 
                double tolerance);
std::vector<double> proposeDirch(const std::vector<double>& mean, double denom,
                                 double tolerance);
double logQDirch(const std::vector<double>& newMean, 
                 const std::vector<double>& oldMean, 
                 double denom, double tolerance);
double proposeNorm(double mean, double variance);
double logQNorm(double newMean, double oldMean, double variance);
double safeBetaPar(double x);
void safeDirchPar(std::vector<double>& x);

/// \class ParameterBase
/// \brief Pure virtual class provide interface for Parameter
///
class ParameterBase {
public:
  ParameterBase(std::string label);
  virtual ~ParameterBase(void);

  virtual void Adapt(const double accept);
  void Assign(const boost::any& value);
  std::string Label(void) const;
  void SetLabel(const std::string& label);
  const boost::any Value(void) const;

private:
  std::string label_;

};

/// \class ParameterT
/// \brief Base class for model parameters
///
/// ParameterT is one of the three workhorses in the simulation framework.
/// Each parameter in the statistical model must be derived separately from
/// ParameterT. It is not necessary to declare and define separate classes
/// for parameters that share the same probabilistic structure, e.g., 
/// allele counts in different populations or measurements of yields in
/// different experimental blocks, but each parameter must be pushed
/// separately onto the step_ stack in Model.
///
/// It is often useful for the derived class to store a pointer to the
/// model of which it is part. This allows the Model class to define access
/// functions to the values of other parameters, facilitating calculation
/// of the full conditionals.
///
/// lPrior() and llike(), the log prior and log likelihood for a particular 
/// parameter respectively, are used only as a sum. Thus, if it is more
/// convenient to write the full conditional in a single function, either
/// may be used. I typically find it easiest to write lPrior() as expressing
/// the "probability" of the parameter, given hyperparameters on which it
/// depends and llike() as expressing the "probability" of parameters (or 
/// data that depend on this one. But your mileage may vary.
///
template <typename T>
class ParameterT : public ParameterBase {
public:
  /// Constructor
  ///
  /// \param label    a string identifier for the parameter
  ///
  ParameterT(std::string label)
    : ParameterBase(label)
  {}

  /// Destructor
  ///
  virtual ~ParameterT(void) {}

public:
  // must be public to allow access from Step classes
  // cannot be pure virtual, because we want 0 returns unless they
  // are overridden

  /// Returns likelihood associated with current parameter
  ///
  /// \param x  Value of the parameter in the current iteration
  ///
  virtual double llike(const T x) const {
    return 0.0;
  }

  /// Returns prior associated with current parameter
  ///
  /// \param x  Value of the parameter in the current iteration
  ///
  virtual double lPrior(const T x) const {
    return 0.0;
  }

  /// Propose a new value for a parameter in an M-H step
  ///
  /// \param current  Value of the parameter in the current iteration
  ///
  virtual T propose(const T current) const {
    return current;
  }

  /// Probability of proposing x, given starting from y
  ///
  /// \param x
  /// \param y
  ///
  virtual double lQ(const T x, const T y) const {
    return 0.0;
  }

  /// Assign value
  ///
  /// \param x  Value to assign to this parameter
  void Assign(const T& x) {
    x_ = x;
  }

  /// Function used to update a deterministic node
  ///
  virtual const T Function(const bool doCalc = true) const {
    return x_;
  }

  /// Value of the parameter
  ///
  virtual const T Value(void) const {
    return x_;
  }

private:
  T x_;

};


/// \class StepBase
/// \brief Base class for steps associated with different parameter types
///
class StepBase {
public:
  StepBase(ParameterBase* parameter, unsigned long accept, 
           unsigned long ct, const double w, const double m, 
           const double lowBound, const double highBound);
  virtual ~StepBase(void);

  /// Select new parameter from sampler
  ///
  virtual void DoStep(void) = 0;

  /// Set bounds on parameter value
  ///
  /// \param low      lower bound
  /// \param high     upper bound
  ///
  virtual void SetBounds(double low, double high) = 0;

  /// Set width of step in slice sampler
  ///
  /// \param w        width
  ///
  virtual void SetW(double w) = 0;

  /// Set maximum number of steps in slice sampler
  ///
  /// \param m        maximum number of steps allowed
  ///
  virtual void SetM(int m) = 0;

  /// Returns parameter value of the current step
  ///
  const boost::any aValue(void) const;
  const double Value(void) const;
  const int iValue(void) const;
  const std::vector<double> dVecValue(void) const;
  const std::vector<int> iVecValue(void) const;

  int accept(void) const;
  std::string Label(void) const;
  ParameterBase* Par(void) const;
  void ResetAccept(void);
  void SetLabel(const std::string& label);

protected:
  ParameterBase* par_;          ///< holds the data and methods
  unsigned long accept_;        ///< acceptance count for M-H
  unsigned long ct_;            ///< number of choices so far

  double w_;                    ///< slice width
  double m_;                    ///< maximum number of steps in slice sampler
  double lowBound_;             ///< smallest value of parameter allowed
  double highBound_;            ///< largest value of parameter allowed

};

/// \class Step
/// \brief Base class for all steps associated with one parameter type
///
template <typename T>
class Step : public StepBase {
public:
  /// Constructor
  ///
  /// \param parameter  Pointer to the parameter associated with this step
  ///
  explicit Step(ParameterT<T>* parameter)
    : StepBase(parameter, 0, 0, 0.0, 0.0, 0.0, 0.0), rng_(GetRNG()),
      parT_(parameter)
  {}

  /// Destructor
  ///
  virtual ~Step(void) {}

protected:
  /// Constructor
  ///
  /// \param parameter  Pointer to the parameter associated with this step
  /// \param w          step width for slice sampler
  /// \param m          maximum number of steps for slice sampler
  /// \param lowBound   minimum value of parameter allowed
  /// \param highBound  maximum value of parameter allowed
  ///
  Step(ParameterT<T>* parameter, double w, double m, double lowBound,
       double highBound)
    : StepBase(parameter, 0, 0, w, m, lowBound, highBound), rng_(GetRNG())
  {}

  lot& rng_;             ///< shared random number generator
  ParameterT<T>* parT_;  ///< pointer to parameter associated with this step

};

/// \class MetroStepT
/// \brief Implements Metropolis-Hastings step for a parameter
///
template <typename T>
class MetroStepT : public Step<T> {
public:
  /// Constructor
  ///
  /// \param parameter  Pointer to the parameter associated with this step
  ///
  explicit MetroStepT(ParameterT<T>* parameter)
    : Step<T>(parameter), met_(parameter), 
      llike_(&ParameterT<T>::llike), lPrior_(&ParameterT<T>::lPrior),
      propose_(&ParameterT<T>::propose), lQ_(&ParameterT<T>::lQ)
  {}

protected:
  /// Accept or reject
  ///
  /// \param newX  Proposed value
  /// \param oldX  Old value
  ///
  /// \returns true  for accept
  /// \returns false for reject
  ///
  bool Accept(T& newX, T& oldX) {
    double piNew = (met_->*llike_)(newX) + (met_->*lPrior_)(newX);  
    double piOld = (met_->*llike_)(oldX) + (met_->*lPrior_)(oldX);
    double qNew = (met_->*lQ_)(newX, oldX);
    double qOld = (met_->*lQ_)(oldX, newX);
    //          log((pi(Y)q(X|Y))/(pi(X)q(Y|X)))
    double la = piNew + qOld - piOld - qNew;
    if (Step<T>::rng_.uniform() < exp(la)) {
      return true;
    } else {
      return false;
    }
  }

  /// Assign value to parameter associated with this step
  ///
  /// \param x  The value to assign
  ///
  void Assign(const T& x) {
    met_->Assign(x);
  }

  /// Get a new value from an M-H step
  ///
  virtual void DoStep(void) {
    T val = boost::any_cast<T>(met_->Value());
    T xTry = (met_->*propose_)(val);
    if (Accept(xTry, val)) {
      ++Step<T>::accept_;
      met_->Assign(xTry);
    } else {
      met_->Assign(val);
    }
  }

protected:
  ParameterT<T>* met_;   ///< pointer to parameter associated with this step

private:
  // no reason to call these

  /// Value of parameter associated with this step
  ///
  const T Value(void) const {
    return met_->Value();
  }

  /// Set bounds on parameter value 
  ///
  /// \param low    lower bound
  /// \param high   upper bound
  ///
  virtual void SetBounds(double low, double high) 
  {}

  /// Set width of step for slice sampler (has no effect in MetroStep)
  ///
  /// \param w    width
  ///
  virtual void SetW(double w) 
  {}

  /// Set number of steps allowed in slice sampler (has no effect in MetroStep)
  ///
  /// \param m    maximum number of steps allowed
  ///
  virtual void SetM(int m) 
  {}

  /// typedef for convenient access to function pointer
  ///
  typedef double(ParameterT<T>::*llikePtr)(const T) const;
  /// typedef for convenient access to function pointer
  ///
  typedef double(ParameterT<T>::*lPriorPtr)(const T) const;
  /// typedef for convenient access to function pointer
  ///
  typedef T(ParameterT<T>::*proposePtr)(const T) const;
  /// typedef for convenient access to function pointer
  ///
  typedef double(ParameterT<T>::*lQPtr)(const T, const T) const;

  const llikePtr llike_;        ///< pointer to likelihood
  const lPriorPtr lPrior_;      ///< pointer to prior
  const proposePtr propose_;    ///< pointer to proposal
  const lQPtr lQ_;              ///< pointer to q()

};

/// \class AdaptMetroStepT
/// \brief Implements Metropolis-Hastings step for a parameter
///
/// This version allows an adaptive phase to be included in during the
/// burnin
///
template <typename T>
class AdaptMetroStepT : public MetroStepT<T> {
public:
  /// Constructor
  ///
  /// \param parameter  Pointer to the parameter associated with this step
  /// \param nBurnin    Number of iterations in burn in
  /// \param adapt      Number of iterations for adaptation
  /// \param interval   Number of iterations between adaptive adjustments
  ///
  explicit AdaptMetroStepT(ParameterT<T>* parameter, 
                 const unsigned long nBurnin, 
                 const unsigned long adapt,
                 const unsigned long interval)
    : MetroStepT<T>(parameter),
      adapt_((nBurnin < 5000) ? 0: adapt),
      interval_(interval), ct_(0)
  {}

  /// Simple modification to allow adaptive phase of M-H sampling
  ///
  void DoStep(void) {
    T current = MetroStepT<T>::met_->Value();
    T xTry = MetroStepT<T>::met_->propose(current);
    if (Accept(xTry, current)) {
      ++Step<T>::accept_;
      MetroStepT<T>::met_->Assign(xTry);
    } else {
      MetroStepT<T>::met_->Assign(current);
    }
    ++ct_;
    if ((ct_ < adapt_) && ((ct_ % interval_) == 0)) {
      MetroStepT<T>::met_->Adapt(static_cast<double>(Step<T>::accept_)/ct_);
    }
  }

private:
  /// Set bounds on parameter value 
  ///
  /// \param low    lower bound
  /// \param high   upper bound
  ///
  virtual void SetBounds(double low, double high) 
  {}

  /// Set width of step for slice sampler (has no effect in MetroStep)
  ///
  /// \param w    width
  ///
  virtual void SetW(double w) 
  {}

  /// Set number of steps allowed in slice sampler (has no effect in MetroStep)
  ///
  /// \param m    maximum number of steps allowed
  ///
  virtual void SetM(int m) 
  {}

  const unsigned long adapt_;        // number of iterations for adaptive phase
  const unsigned long interval_;     // interval for adjusting M-H parameters
  unsigned long ct_;

};


/// \class SliceStep
/// \brief Implements a slice sampler
///
/// Note: only univariate slices are supported
///
/// ParameterT constructors must provide appropriate step width,
/// parameter->W(), and number of tries, parameter->M(), if defaults 
/// (1 and MUnbounded, respectively) are to be avoided.
/// lowBound_ and highBound_ defaults must be reset if parameter range is
/// bounded, e.g., frequencies in [0,1].
///
class SliceStep : public Step<double> {
public:
  explicit SliceStep(ParameterT<double>* parameter);

  void SetBounds(double low, double high);
  void SetW(double w);
  void SetM(int m);

protected:
  virtual void DoStep(void);

private:
  void Assign(const double x);
  void SetSlice(const double x);
  double Sample(const double x);
  const double Value(void) const; 

  double r_;                    // lower limit of slice
  double l_;                    // upper limit of slice
  double z_;
  unsigned long ct_;            // number of choices so far

  static const double DefaultW;
  static const long MUnbounded;

  typedef double(ParameterT<double>::*llikePtr)(const double) const;
  typedef double(ParameterT<double>::*lPriorPtr)(const double) const;

  ParameterT<double>* dbl_;
  llikePtr llike_;
  lPriorPtr lPrior_;

};

/// \class FunctionStepT
/// \brief Implements a deterministic node
///
/// Often parameters of interest are deterministic functions of other
/// statistical parameters. A FunctionStep allows us to express that
/// relationship. DoStep() simply invokes the Function() associated with
/// this parameter and returns the result.
///
template<typename T>
class FunctionStepT : public Step<T> {
public:
  /// Constructor
  ///
  /// \param parameter  Pointer to parameter associated with this step
  ///
  explicit FunctionStepT(ParameterT<T>* parameter)
    : Step<T>(parameter), func_(parameter)
  {}

  /// Return the value associated with this node
  ///
  /// The value is returned by the Function() associated with this paramter.
  ///
  const T Value(void) {
    return func_->Function();
  }

private:
  /// Return the value associated with this node
  ///
  /// The value is returned by the Function() associated with this paramter.
  ///
  void DoStep(void) {
    func_->Assign(func_->Function());
  }

  // no reason to call these

  /// Sets bounds for the parameter (empty for FunctionStep)
  ///
  /// \param low     lower bound
  /// \param high    upper bound
  ///
  void SetBounds(const double low, const double high) 
  {}

  /// Sets width for slice sampler (empty for FunctionStep)
  ///
  /// \param w       step width
  ///
  void SetW(const double w) 
  {}

  /// Sets maximum number of steps for slice sampler (empty for FunctionStep)
  ///
  /// \param m       maximum number of steps allowed
  ///
  void SetM(const int m)
  {}

  ParameterT<T>* func_;

};

typedef std::vector<StepBase*> ModelSteps;///< pointer to each Step in the Model
typedef std::vector<boost::any> SampleVector;   ///< parameters at one iteration
typedef SampleVector::const_iterator SampleIter; ///< paramter iterator
typedef std::vector<SampleVector> Results; ///< SampleVectors for all iterations
typedef Results::const_iterator ResultsIter; ///< iterator over iterations

/// \class Model
/// \brief Implements the statistical model
///
/// To build a model, derive a new class from Model. In its constructor,
/// push Steps of the appropriate type onto step_. If you're happy with
/// the default reports, the only other thing you'll need to do is to provide
/// accessor functions to values of the parameters in step_ (or make them
/// public so that Parameters can access them directly).
/// 
/// If you want to calculate DIC for the model, you'll have to provide an
/// appropriate override for Llike(), but everything else will be taken
/// care of automatically.
///
class Model {
public:
  virtual ~Model(void);

  void Simulation(std::ostream& outf, bool interimReport);
  virtual void Summarize(int i, std::ostream& outf);
  std::string Label(int i) const;
  void ReportDic(std::ostream& outf);

  // must be public to allow overrides for specific models
  // cannot be pure virtual, because the default is reasonable in
  // most cases
  virtual void Record(const SampleVector& p);
  virtual void InterimReport(std::ostream& outf, std::string header, 
                             int progress, int goal);
  virtual void ReportHead(std::ostream& outf);

  virtual void Report(std::ostream& outf, const int lastPar = -1);
  virtual double Llike(const SampleVector& par) const;
  void SetBurnIn(const int nBurnin);
  void SetSample(const int nSample);
  void SetThin(const int thin);

protected:
  Model(int nBurnin, int nSample, int nThin, bool calc = false, 
        bool useMedian = false);

  double percent(int k, int n) const;
  double percent(const std::vector<int>& x, int n) const;
  double percent(const std::vector<std::vector<int> >& x, int n) const;

  int nBurnin_;                 ///< number of iterations for burn in
  int nSample_;                 ///< number of iterations for sample
  int nThin_;                   ///< number of iterations between saving results
  unsigned nElem_;              ///< number of parameters in the model
  SimpleStatistic likelihood_;  ///< stores likelihood for DIC calculations

  ModelSteps step_;             ///< vector of parameters to sample
  Results results_;             ///< vector (boost::any) of stored results
  virtual SampleVector Parameters(void) const;

  boost::format* summaryFormat_; ///< boost::format for summary output

private:
  void Reset(void);
  void Generation(void);

  boost::progress_display* pd_;
  bool calculateLikelihood_;
  bool useMedian_;

};

/// typedef for convenienc access to double paramters
///
typedef ParameterT<double> Parameter;
/// typedef for convenienc access to double paramters
///
typedef FunctionStepT<double> FunctionStep;
/// typedef for convenienc access to double paramters
///
typedef MetroStepT<double> MetroStep;

#endif

// Local Variables: //
// mode: c++ //
// End: //
