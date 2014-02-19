/*! \mainpage

    <table>
        <tr><th>Library     <td>Scythe MCMC - A Scythe Markov Chain Monte Carlo C++ Framework
        <tr><th>Author      <td>Tristan Zajonc (tristanz@gmail.com)
        <tr><th>Source      <td>http://ksghome.harvard.edu/~zajonct/
        <tr><th>Version     <td>0.1
    </table>

    \section intro INTRODUCTION

    Scythe MCMC is a C++ header library that eases the development of Markov Chain Monte Carlo (MCMC).  It is based on the
    <a href="http://scythe.wustl.edu/">Scythe Statistical Library</a>.
    Scythe MCMC provides an execution framework, including command line and option file parsing, that reduces the amount of boilerplate
    code needed to write custom MCMC routines.  It also provides common MCMC step types, including Gibbs, Metropolis-Hastings,
    and Slice sampling.  Users can experiment with which sampling steps provide the best results and implement their own sampling
    steps as desired.

    Scythe MCMC was motivated with the need to combine simpler samplers, such as univariate slice sampling, with more specialized
    samplers for particular problems.  Future versions of Scythe MCMC will include common samplers for nonparametric Bayesian models,
    including a Collapsed Gibbs sampler for the Dirichlet Process, and Slice samplers for the Dirichlet Process and Indian Buffet Process.

    There are many choices for writing MCMC samplers.  Scythe MCMC is similar to, and heavily inspired by,
    <a href="http://darwin.eeb.uconn.edu/mcmc++/mcmc++.html">MCMC++</a>, but is based
    on the Scythe, which provides convenient matrix types, random number generators,
    and probability distributions. Scythe MCMC is different in focus from
    <a href="http://mcmcpack.wustl.edu/wiki/index.php/Main_Page">MCMCPack</a>, which
    is also based on Scythe and interfaces with R.  Users wishing to distribute a specific model to R users
    should consider contributing to MCMCPack directly. Unlike MCMCPack, Scythe MCMC makes no attempt to interface with R
    and assumes samplers are executed from the command line (although this is not required).
    It also provides common algorithms (Gibbs, Metropolis-Hastings, Slice Sampling) that ease
    development of samplers for many models.

    Even though implementing models in ScytheMCMC requires more work than
    implementing comparable models in <a href="http://mathstat.helsinki.fi/openbugs/">Bugs</a>
    or <a href="http://www-ice.iarc.fr/~martyn/software/jags/">JAGS</a>, Scythe MCMC generally
    leads to faster samplers that can exploit the particular structure of a given problem.  It also
    allows development of models that cannot be sampled effectively using either Bugs or Jags, such
    nonparametric Bayesian models without truncation approximations.

    \section features FEATURES

    - Based on the <a href="http://scythe.wustl.edu/">Scythe Statistical Library</a>.
    - Eliminates commandline and option parsing boilerplate code using
      <a href ="http://code.jellycan.com/simpleopt/">SimpleOpt</a> and <a href="http://code.jellycan.com/simpleini/">SimpleIni</a>.
    - Built in sampling steps:
        - Gibbs,
        - Metropolis-Hastings,
        - Slice Sampling.
        - Walker (2008) sampler for DP.
    - Nonparameteric samplers including:
       - Collapsed gibbs samplers for Dirichlet Process,
       - Slice sampler for Dirichlet Process and Indian Buffet Process.
    - MIT License.

    \section gettingstarted GETTING STARTED


    \section license MIT LICENSE

    The license text below is the boilerplate "MIT License" used from:
    http://www.opensource.org/licenses/mit-license.php

    Copyright (c) 2009, Tristan Zajonc

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is furnished
    to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
    FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
    COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
    IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/progress.hpp>
#include <boost/timer.hpp>
#include <boost/algorithm/string.hpp>
#include <ctime>
#include <iostream>
#include <fstream>
#include <scythestat/rng/mersenne.h>
#include <scythestat/rng/lecuyer.h>
#include <scythestat/distributions.h>
#include <scythestat/matrix.h>
#include <scythestat/rng.h>
#include <scythestat/smath.h>
#include <scythestat/stat.h>
#include <scythestat/ide.h>
#include <scythestat/defs.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
//--S [] 2014/02/19 : Sang-Wook Lee
//#include "Simple/SimpleOpt.h"
//#include "Simple/SimpleIni.h"
//--E [] 2014/02/19 : Sang-Wook Lee
// #include <mkl_vml_functions.h>
// #include <mkl_vml_defines.h>

//--S [] 2014/02/19 : Sang-Wook Lee
//#include "scythe_extensions.h"
//--E [] 2014/02/19 : Sang-Wook Lee

/// Scythe MCMC version.
#define SCYTHE_MCMC_VERSION "0.1"
/// Value of missing data. If data is stored as double, use round(x)==MISSING.
#define MISSING -9999

using namespace scythe;
using namespace std;

/// Scythe column-oriented double matrix typedef.
typedef Matrix<double, Col> matrix;
/// Scythe column-oriented integer matrix typedef.
typedef Matrix<int, Col> imatrix;
/// Pair typedef for integers.  Used to store (x, y) coordinates of a matrix location.
typedef pair<int, int> ipair;

/// Global positive infinity representation.
double dInf = std::numeric_limits<double>::infinity();
double iInf = std::numeric_limits<int>::infinity();

/// Global L'Ecuyer random number generator.
lecuyer myrng;
//mersenne myrng;

/*! \brief Basic MCMC options.
 *
 * These options are generally specified at the command line and are generic to all
 * MCMC samplers.  Other options are passed through an .ini config file.
 */
struct MCMCOptions {
  /// Sample size. (iterations - burnin)/(thin + 1).
  int sample_size;
  /// Thinning interval
  int thin;
  /// Burn in period.
  int burnin;
  /// Chains.
  int chains;
  /// Seed for random number generator.
  unsigned long random_seed [6];
  /// Config file (.ini style).
  string config_file;
  /// Out file to save parameter chains.
  string out_file;
};
/// Global MCMCOptions object.  It is convenient to store command line options globally to avoid passing them
/// around everwhere.
MCMCOptions mcmc_options;

/*! \brief Check if file exists.
 *
  \param strFilename Filename to check.
  \returns True is file exists. False otherwise.
  \note Source: http://www.techbytes.ca/techbyte103.html.
 */
bool FileExists(string strFilename) {
  struct stat stFileInfo;
  bool blnReturn;
  int intStat;

  // Attempt to get the file attributes
  intStat = stat(strFilename.c_str(), &stFileInfo);
  if (intStat == 0) {
    // We were able to get the file attributes
    // so the file obviously exists.
    blnReturn = true;
  } else {
    // We were not able to get the file attributes.
    // This may mean that we don't have permission to
    // access the folder which contains this file. If you
    // need to do that level of checking, lookup the
    // return values of stat which will give you
    // more details on why stat failed.
    blnReturn = false;
  }

  return(blnReturn);
}

/*! \brief Convert any streaming type to CSV.
 *
 */
template <class T>
inline string to_csv (const T& t) {
  std::stringstream ss;
  ss << t;
  string s = ss.str();
  replace(s.begin(), s.end(), ' ', ',');
  return s;
}

/*! \brief Convert any streaming type to string
 *
 */
template <class T>
inline string to_string (const T& t) {
  std::stringstream ss;
  ss << t;
  string s = ss.str();
  return s;
}


/*! \brief Shows command line usage.

   Diplays:
<pre>
   ScytheMCMC v. 0.1

     Usage:
       --help                       displays help message
       --config-file=filename       config file name
       --out-file=filename          out file name
       --chains=arg                 number of chains
       --sample-size=arg            retained sample size
       --burnin=arg                 burn in period
       --thin=arg                   thinning iterval (1 = no thinning)
       --random-seed=arg            random number seed (0 uses current timestamp)

     Example: ./command --config-file=command.ini --out-file=out.txt --chains=1 --sample-size=1000
              --burnin=0 --thin=1 --random-seed=0
</pre>
*/
void ShowUsage() {
  cout << "  Usage:" << endl
       << "    --help \t\t\t displays help message" << endl
       << "    --config-file=filename \t config file name" << endl
       << "    --out-file=filename \t out file name" << endl
       << "    --chains=arg \t\t number of chains" << endl
       << "    --sample-size=arg \t\t retained sample size" << endl
       << "    --burnin=arg \t\t burn in period" << endl
       << "    --thin=arg \t\t\t thinning iterval (1 = no thinning)" << endl
       << "    --random-seed=arg \t\t random number seed (0 uses current timestamp, 1 attempts to load from lecuyer.seed file)" << endl << endl;

  cout << "  Example: ./command --config-file=command.ini --out-file=out.txt --chains=1 --sample-size=1000" << endl
       << "\t   --burnin=0 --thin=1 --random-seed=0" << endl << endl;
}

/*! \brief Parses command line options.

  Parses and stores command line options and stores them in MCMCOptions instance
  mcmc_options.  Initializes random mersenne instance myrng.
  \see ShowUsage.
  \see mcmc_options.
  \see myrng.
*/
//--S [] 2014/02/19 : Sang-Wook Lee
/*
void DefaultStartUp(int argc, char* argv[]) {
  cout << "ScytheMCMC v. " << SCYTHE_MCMC_VERSION << endl << endl;

  // Config parser is based on SimpleOpt Library.
  // See: http://code.jellycan.com/simpleopt/
  enum { OPT_SAMPLE_SIZE, OPT_THIN, OPT_BURNIN, OPT_CHAINS, OPT_RANDOM_SEED, OPT_CONFIG_FILE, OPT_OUT_FILE, OPT_HELP };
  CSimpleOpt::SOption g_rgOptions[] = {
    { OPT_HELP, "--help", SO_NONE },
    { OPT_CONFIG_FILE, "--config-file", SO_REQ_CMB },
    { OPT_OUT_FILE, "--out-file", SO_REQ_CMB },
    { OPT_CHAINS, "--chains", SO_REQ_CMB },
    { OPT_SAMPLE_SIZE, "--sample-size", SO_REQ_CMB },
    { OPT_BURNIN, "--burnin", SO_REQ_CMB },
    { OPT_THIN, "--thin", SO_REQ_CMB },
    { OPT_RANDOM_SEED, "--random-seed", SO_REQ_CMB },
    SO_END_OF_OPTIONS
  };
  CSimpleOpt args(argc, argv, g_rgOptions);

  int setflags = 0;
  while (args.Next()) {
    if (args.LastError() == SO_SUCCESS) {
      switch (args.OptionId()) {
      case OPT_HELP:
        ShowUsage();
        exit(0);
        break;
      case OPT_CONFIG_FILE:
        setflags++;
        mcmc_options.config_file = args.OptionArg();
        break;
      case OPT_OUT_FILE:
        setflags++;
        mcmc_options.out_file = args.OptionArg();
        break;
      case OPT_CHAINS:
        setflags++;
        mcmc_options.chains = atoi(args.OptionArg());
        break;
      case OPT_SAMPLE_SIZE:
        setflags++;
        mcmc_options.sample_size = atoi(args.OptionArg());
        break;
      case OPT_BURNIN:
        setflags++;
        mcmc_options.burnin = atoi(args.OptionArg());
        break;
      case OPT_THIN:
        setflags++;
        mcmc_options.thin = atoi(args.OptionArg());
        break;
      case OPT_RANDOM_SEED:
        setflags++;
        unsigned long seed = atoi(args.OptionArg());
				cout << "TEST" << seed<<endl;
				if(seed == 0) {
					seed = time(NULL);
					for(int i = 0; i < 6; ++i) {
						mcmc_options.random_seed[i] = seed;
					}
				}
				else if (seed == 1) {
					// try to load seed from file
					cout << "Try loading file..." << endl;
					if (FileExists("lecuyer.seed")) {
						Matrix<unsigned long, Col> seedmat("lecuyer.seed");
						for(int i = 0; i < 6; ++i) {
							mcmc_options.random_seed[i] = seedmat(i);
						}
					}
					else {
						cout << "No seed exists..." << endl;
						seed = time(NULL);
						for(int i = 0; i < 6; ++i) {
							mcmc_options.random_seed[i] = seed;
						}
					}
				}
				else {
					for(int i = 0; i < 6; ++i) {
						mcmc_options.random_seed[i] = seed;
					}
				}
        break;
      }
    } else {
      cout << "Invalid argument: " << args.OptionText() << endl;
      exit(1);
    }
  }

  // Check that all required flags are set.
  if (setflags != 7) {
    ShowUsage();
    exit(1);
  }

  // Check argument values.
  if (mcmc_options.sample_size <= 0) {
    cout << "Sample size must be positive.\n";
    exit(1);
  }
  if (mcmc_options.burnin < 0) {
    cout << "Burn in cannot be negative.\n";
    exit(1);
  }
  if (mcmc_options.thin < 1) {
    cout << "Thinning interval must be positive.\n";
    exit(1);
  }
  if (!FileExists(mcmc_options.config_file)) {
    cout << "Config file does not exist.\n";
    exit(1);
  }

  // Seed random number generator.
  //myrng.initialize(mcmc_options.random_seed);
	myrng.SetSeed(mcmc_options.random_seed);
	
  // Print mcmc options.
  cout << "Arguments:" << endl;
  cout << "  Config file: " << mcmc_options.config_file << endl
       << "  Chains: " << mcmc_options.chains << endl
       << "  Desired sample size: " << mcmc_options.sample_size << endl
       << "  Burn in period: " << mcmc_options.burnin << endl
       << "  Thinning interval: " << mcmc_options.thin << endl
       << "  Random number seed: "
 						<< mcmc_options.random_seed[0] << " "
 						<< mcmc_options.random_seed[1] << " "
						<< mcmc_options.random_seed[2] << " "
						<< mcmc_options.random_seed[3] << " "
						<< mcmc_options.random_seed[4] << " "
						<< mcmc_options.random_seed[5] << endl << endl;
}
*/
//--E [] 2014/02/19 : Sang-Wook Lee


/*! \brief Main parameter class.
 *
 * The model is defined as many instances of parameter subclasses.  Different Step types require different methods
 * to be implemented in each Parameter subclass.
 *
 * \section overwrite Methods that should be implemented in subclass:
 * Required for all step types:
 *   - Constructor that sets, at least, Parameter::track_ and Parameter::name_
 *   - Parameter::Value
 *   - Parameter::Save
 *
 * GibbsStep:
 *   - Parameter::StartingValue
 *   - Parameter::RandomPosterior
 *
 * Metropolis-Hastings:
 *   - Parameter::StartingValue
 *   - Parameter::LogDensity
 *
 * Slice:
 *   - Parameter::StartingValue
 *   - Parameter::LogDensity
 *
 * FunctionStep (deterministic):
 *   - Parameter::Function
 *
 * \section examples Examples:
 *
 * \subsection example1 Normal mean parameter with normal prior, known standard deviation:
 *
 \code
 // Priors
 double mean_mu = 0;
 double mean_sigma = 2;

 // Parameters
 double mean = 0; // free parameter
 double sd = 1; // known

 // Data
 matrix X('X.txt');

 // Define the mean parameter, implementing methods required for GibbsStep.
 class MeanParameter : public Parameter<double> {
 public:
   /// Default constructors that call base constructors. (REQUIRED)
   MeanParameter() : Parameter<double>() {}
   MeanParameter(bool track, string name) : Parameter<double>(track, name) {}

  /// Draw a random value from analytical posterior for GibbsStep.
  double RandomPosterior() {
     // Posterior of mean with known variance, (Gelman et al (2004) p 49.)
     double x_mean = mean(X);
     int n = X.rows();
     double denom = 1/(1/pow(mean_sigma,2) + n / pow(sd,2))
     double posterior_mean = ((1/pow(mean_sigma,2)) * mean_mu + (n / pow(sd,2)) * x_mean) / denom
     double posterior_sd = sqrt(denom);
     return myrng.rnorm(posterior_mean, posterior_sd);
   }

   /// Draw starting value from prior.
   double StartingValue() {
     return myrng.rnorm(mean_mu, mean_sigma);
   }

   /// Save back to global location
   void Save(double new_value) {
     mean = new_value;
   }

   /// Return value from global location.
   double Value() {
     return mean;
   }
 };
 \endcode
 */
template<typename ParameterStorageType>
class Parameter {
public:
  /// Default class constructor.
  Parameter() {}
  /*! \brief Base class constructor
   *
   * \param track True if parameter should be tracked and saved.
   * \param label Label of parameter for tracking purposes.
  */
  Parameter(bool track, string label) : track_(track), label_(label) {}

  /// Function for deterministic nodes.
  /// \return Function value.
  virtual ParameterStorageType Function() {
    return 0.0;
  };

  /// Starting value.
  /// \return Starting value.
  virtual ParameterStorageType StartingValue() {
    return 0.0;
  };

  /// Log of the probability density (plus constant)
  /// \param value Value of parameter to evaluate density at.
  /// \return Log of probability density (plus constant)
  virtual double LogDensity(double value) {
    return 0.0;
  }

  /// Return a random draw from the posterior.
  /// Random draw from posterior is called by GibbsStep.
  /// \return Random draw from posterior.
  virtual ParameterStorageType RandomPosterior() {
    return 0.0;
  }

  /// Value of parameter.
  /// \return Parameter value
  virtual ParameterStorageType Value() {
    return 0.0;
  };

  /// Save a new value of the parameter.
  /// \param new_value New value to save.
  virtual void Save(ParameterStorageType new_value) {};

  /// Parameter is tracked / saved.
  /// \return True if parameter is tracked.
  bool Track() {
    return track_;
  }

  /// String label of parameter.
  /// \return Label of parameter, for purposes of saving output.
  string Label() {
    return label_;
  }
private:
  /// Should this variable be tracked?
  bool track_;
  /// Name of variable for tracking purposes.
  string label_;
};

/*! \defgroup steps Implemented MCMC Step Types
 *
 * MCMC algorithms typically consist of many steps.  While users will often want
 * to implement their own step types, Scythe MCMC has implemented several that arise
 * over and over.  MetropStep and SliceStep in particular only require specification of
 * a log density function (minus an unknown constant).
 *
 */

/*! \brief Base step class.
 *
 * \ingroup steps
 *
 * The step (Gibbs, Metropolis-Hastings, Slice, etc) describes how the sampler
 * calculates the next value of each parameter.  The key function is Step::DoStep, which is
 * equivalent to the Execute function in a <a href="http://en.wikipedia.org/wiki/Command_pattern">Command Pattern</a>.
 * Samplers consists of a many of steps.
 *
 * \see GibbsStep
 * \see FunctionStep
 * \see MetropStep
 * \see SliceStep
 * \see Sampler
 */
class Step {
public:
  /*! \brief Take step.
   *
   * Executes the main step function, such as taking a MH step or deterministic step.
   */
   virtual void DoStep() {}
  /*! \brief Set starting value.
   *
   *  Calls the parameter's Parameter::Starting function and then saves the result using Parameter::Save.
   */
  virtual void Start() {}

  /*! \brief Return parameter value
   *
   * \return String representation of parameter value for csv output.
   * \see Parameter::Value
   */
  virtual string ParameterValue() {
    return "";
  }

  /*! \brief Return parameter label
   *
   * \return The label of the parameter associated with the step instance.
   * \see Parameter::Label
   */
  virtual string ParameterLabel() {
    return "";
  }

  /*! Return the tracking status
   *
   * \return The tracking status of the parameter associated with the step instance.
   * \see Parameter::Track
   */
  virtual bool ParameterTrack() {
    return true;
  }
};

/*! \brief Gibbs step.
 *
 * \ingroup steps
 *
 * A univariate Gibbs step draws from the Parameter::RandomPosterior function of a Parameter and then saves the result
 * using Parameter:Save.
 */
template <typename ParameterType, typename ParameterStorageType>
class GibbsStep: public Step {
public:
  /// Default constructor, for copy assignments, etc.
  GibbsStep() {}
  /*! \brief Main constructor taking a Parameter object.
   *  \param parameter A Parameter that implements Parameter::RandomPosterior.
   */
  GibbsStep(ParameterType parameter) : parameter_(parameter) {
  }
  /*! \brief Take a Gibbs step for the parameter.
   *
   *  Draws from the parameter's Parameter::RandomPosterior function and then saves the result using Parameter::Save.
   */
  void DoStep() {
		parameter_.Save(parameter_.RandomPosterior());
  }

  void Start() {
    parameter_.Save(parameter_.StartingValue());
  }

  string ParameterLabel() {
    return parameter_.Label();
  }

  string ParameterValue() {
    return to_csv<ParameterStorageType>(parameter_.Value());
  }

  bool ParameterTrack() {
    return parameter_.Track();
  }
private:
  /// Parameter associated with step instance.
  ParameterType parameter_;
};

/*! \brief Deterministic function step.
 *
 * \ingroup steps
 *
 * Deterministic nodes can be helpful to track summary statistics or to reduce computations through intermediate use
 * of sufficient statistics across multiple parameters.  FunctionStep calls the parameters Parameter::Function method
 * and saves the result using Parameter::Save.
 */
template <typename ParameterType, typename ParameterStorageType>
class FunctionStep: public Step {
public:
  /// Default constructor, for copy assignments, etc.
  FunctionStep() {}
  /*! \brief Main constructor taking a Parameter object.
   *
   *  \param parameter A Parameter that implements Parameter::Function.
   */
  FunctionStep(ParameterType parameter) : parameter_(parameter) {
  }
  /*! \brief Take a function step for the parameter.
   *
   *  Calculates the value of Parameter::Function and then saves the result using Parameter::Save.
   */
  void DoStep() {
    parameter_.Save(parameter_.Function());
  }

  /*! \brief Set function step starting value.
   *
   */
  void Start() {
		parameter_.StartingValue();
  }

  string ParameterLabel() {
    return parameter_.Label();
  }

  string ParameterValue() {
    return to_csv<ParameterStorageType>(parameter_.Value());
  }

  bool ParameterTrack() {
    return parameter_.Track();
  }
private:
  /// Parameter associated with step instance.
  ParameterType parameter_;
};

/*! \defgroup proposals Metropolis-Hastings Proposals
 *
 * The MetropStep Metropolis-Hastings class requires both a Parameter instance
 * and a Proposal instance.  Proposal instances define a Draw and LogDensity method.
 * The constructor should set a tuning parameter.
 *
 * MetroStep::DoStep uses the Draw method to draw new proposed values for the parameter
 * and the LogDensity to calculate the proposal terms in the acceptance \f$\alpha\f$.  For
 * symmetric proposal distributions returning 0.0 is enough.
 */

/*! \brief Normal proposal for Metropolis-Hastings.
 *
 *  \ingroup proposals
 *
 *  Normal proposal draws from N(starting_value, standard_deviation).
 */
class NormalProposal {
public:
  NormalProposal() {}
  /*! \brief Constructor
   *
   * \param standard_deviation Tuning parameter for normal proposal.
   */
  NormalProposal(double standard_deviation) : standard_deviation_(standard_deviation) {}

  /*! \brief Draw from normal proposal
   *
   * \param starting_value Starting value.
   */
  double Draw(double starting_value) {
    return myrng.rnorm(starting_value, standard_deviation_);
  }

  /*! \brief Log probability of proposal new value, given starting value.
   *
   * \note Needs to be implemented but can be 0.0 if proposal is symmetric, which it is.
   * \param starting_value Starting value.
   * \param new_value Proposed new value.
   */
  double LogDensity(double new_value, double starting_value) {
    return 0.0; // Symmetric, doesn't matter.
  }

private:
  double standard_deviation_;
};


/*! \brief Beta proposal for Metropolis-Hastings.
 *
 * \ingroup proposals
 *
 * Useful proposal type for parameters with support between 0 and 1.
 *
 * The beta proposal is parameterized using the mean and the inverse denominator.  The denominator
 * is a measure a scale and can be thought of as the number of observed trials bernoulli.  We use the inverse
 * of the denominator so that larger values imply bigger steps, consistent with NormalProposal.  For the Beta distribution
 * the mean is
 * \f[ m = \frac{\alpha}{\alpha + \beta} \f]
 * and the denominator is
 * \f[ d = \alpha + \beta. \f]
 * Therefore
 * \f[ \alpha = m \cdot d \f]
 * and
 * \f[ \beta = d \cdot (1-m). \f]
 *
 */
class BetaProposal {
public:
  /*! \brief Empy constructor */
  BetaProposal() {}
  /*! \brief Main constructor
   *
   * Scale of steps are parameterized using the inverse denominator of a Beta distribution:
   * \f[ \frac{1}{d} = \frac{1}{\alpha + \beta} \f]
   *
   * \param idenominator Tuning parameter equal to the inverse denominator \f$1/d\f$ of a Beta distribution.
   */
  BetaProposal(double idenominator) {
    denom_ = 1 / idenominator;
  }

  /*! \brief Draw from Beta proposal
   *
   * \param starting_value Starting value (mean of Beta distribution).
   */
  double Draw(double starting_value) {
    double alpha = starting_value * denom_;
    double beta = alpha - denom_;
    return myrng.rbeta(alpha, beta);
  }

  /*! \brief Log probability of proposal new value, given starting value.
   *
   * \param starting_value Starting value.
   * \param new_value Proposed new value.
   */
  double LogDensity(double new_value, double starting_value) {
    double alpha = starting_value * denom_;
    double beta = alpha - denom_;
    return dbeta(new_value, alpha, beta);
  }

private:
  double denom_;
};


/*! \brief Log normal proposal for Metropolis-Hastings.
 *
 * \ingroup proposals
 *
 * Useful proposal type for parameters with support between 0 and positive infinity.
 *
 * For convenience, the log normal proposal is parameterized using the unlogged mean (starting value)
 * and the (positive) log standard deviation.  This differs from how scythe::dlnorm is parameterized.
 *
 */
class LogNormalProposal {
public:
  /*! \brief Empy constructor */
  LogNormalProposal() {}
  /*! \brief Main constructor
   *
   * Scale of steps are standard deviation of the logged parameter value:
   *
   * \param logsd Tuning parameter equal to the log standard deviation.
   */
  LogNormalProposal(double logsd) : logsd_(logsd) {}

  /*! \brief Draw from log normal proposal
   *
   * \param starting_value Starting value (unlogged mean of log normal).
   */
  double Draw(double starting_value) {
    double logmean = log(starting_value);
    return myrng.rlnorm(logmean, logsd_);
  }

  /*! \brief Log probability of proposal new value, given starting value.
   *
   * \param starting_value Starting value.
   * \param new_value Proposed new value.
   */
  double LogDensity(double new_value, double starting_value) {
    double logmean = log(starting_value);
    return dlnorm(new_value, logmean, logsd_);
  }

private:
  double logsd_;
};

/*! \brief Metropolis Hastings step
 *
 * \ingroup steps
 *
 * Basic univariate Metropolis-Hastings step.  Requires both a Parameter and Proposal instance.
 */
template <typename ParameterType, typename ProposalType>
class MetropStep: public Step {
public:
  /*! \brief Default constructor, for copy assignments, etc.
   */
  MetropStep() {}
  /*! \brief Main constructor taking a Parameter and Proposal object.
   *
   *  \param parameter A Parameter that implements Parameter::LogDensity
   *  \param proposal A Proposal instance that implements Proposal::LogDensity and Proposal::Draw.
   */
  MetropStep(ParameterType parameter, ProposalType proposal) : parameter_(parameter), proposal_(proposal) {}

  /*! \brief Take a function step for the parameter.
   *
   *  Calculates the value of Parameter::Function and then saves the result using Parameter::Save.
   */
   void DoStep() {
    // Draw a new parameter
    double new_value = proposal_.Draw(parameter_.Value());
    double old_value = parameter_.Value();
    // MH accept/reject criteria
    double alpha = parameter_.LogDensity(new_value) - parameter_.LogDensity(old_value)
                   + proposal_.LogDensity(old_value, new_value) - proposal_.LogDensity(new_value, old_value);
    if (myrng.runif() < min(exp(alpha), 1.0)) {
      parameter_.Save(new_value);
    }
  }

  void Start() {
    parameter_.Save(parameter_.StartingValue());
  }

  string ParameterLabel() {
    return parameter_.Label();
  }

  string ParameterValue() {
    return to_csv<double>(parameter_.Value());
  }

  bool ParameterTrack() {
    return parameter_.Track();
  }
private:
  /// Parameter associated with step instance.
  ParameterType parameter_;
  ProposalType proposal_;
};

/*! \brief Univariate slice sampling step.
 *
 * \ingroup steps
 *
 * Basic univariate slice sampling step with stepping out and shrinkage. Performs a slice sampling update from an initial
 * point to a new point that leaves invariant the distribution with the specifified log density functions.
 *
 * The log desnity function may return -Inf for points outside the support of the distribution.
 * If a lower and/or upper bound is specified for the support, the log density function will not be called outside
 * such limits.
 *
 * \note See Neal, R. M (2003) "Slice Sampling" (with discussion), <i>Annals of Statistics</i>,
 * vol. 31, no. 3, pp. 705-767.
 * \note Code and description based on Neal's R code (March 17, 2008): http://www.cs.toronto.edu/~radford/ftp/slice-R-prog
 * \note With poor initial values or choice of w, slice sampling can get take a very long time to find
 * a suitable slice.  This may appear like a crash.  Experimenting with w may speed sampling, but things tends
 * to work pretty well with default values in many cases.
 */
template <typename ParameterType>
class SliceStep: public Step {
public:
  /// Default constructor, for copy assignments, etc.
  SliceStep() {}
  /*! \brief Main constructor for slice sampling step.
   *
   *  \param parameter A Parameter that implements Parameter::LogDensity
   *  \param w Size of the steps for creating interval (default = 1)
   *  \param lower Lower bound on the support of the distribution (default = -Infinity)
   *  \param upper Upper bound on the support of the distribution (default = Infinity)
   */
  SliceStep(ParameterType parameter, double w, double lower, double upper) :
      parameter_(parameter), w_(w), lower_(lower), upper_(upper) {
		step_count_ = 0;
	}

  /*! \brief Take a slice sampling step for the parameter.
   *
   */
   void DoStep() {
	  // Find the log density at the initial point.
    double x0 = parameter_.Value();
    double gx0 = parameter_.LogDensity(x0);

    // Determine slice level, in log terms
    double logy = gx0 - myrng.rexp(1);

    //Find the initial interval to sample from.
    double u = myrng.runif() * w_;
    double L = x0 - u;
    double R = x0 + (w_ - u);

    // Expand the interval until its ends are outside the slice, or
    // until the limit on steps is reached.

    // allow infinite steps... could hang!
    // FIXME: Add check for too many iterations and exit with error message.
    while (true) {
      if (L <= lower_) {
        break;
      }
      if (parameter_.LogDensity(L) <= logy) {
        break;
      }
      L = L - w_;
    }

    while (true) {
      if (R >= upper_) {
        break;
      }
      if (parameter_.LogDensity(R) <= logy) {
        break;
      }
      R = R + w_;
    }

    // Shrink interval to lower and upper bounds.
    if (L < lower_) {
      L = lower_;
    }
    if (R > upper_) {
      R = upper_;
    }

    // Sample from the interval, shrinking it on each rejection
    double x1, gx1;
    while (true) {
      x1 = myrng.runif() * (R - L) + L;  // Sample between L and R, uniformly.
      gx1 = parameter_.LogDensity(x1);
      if (gx1 >= logy) {
        break;
      }

      if (x1 > x0) {
        R = x1;
      } else {
        L = x1;
      }
    }
		
		// Adapt
		//step_count_++;
		//w_ = (step_count_/(step_count_+1)) * w_ + (1/(step_count_+1))*(R - L);
    // Save the point sampled
    parameter_.Save(x1);
  }

  void Start() {
    parameter_.Save(parameter_.StartingValue());
  }

  string ParameterLabel() {
    return parameter_.Label();
  }

  string ParameterValue() {
    return to_csv<double>(parameter_.Value());
  }

  bool ParameterTrack() {
    return parameter_.Track();
  }
private:
  /// Parameter associated with step instance.
  ParameterType parameter_;
  /// Slice sampling control values
  double w_;
  double lower_;
  double upper_;
	int step_count_;
};

/*! \brief MCMC sampler.
 *
 *  The sampler is the main MCMC object that holds all the MCMC steps for each parameter.  Running the sampler
 *  performs MCMC sampling for the model, saving results, and displaying progress.  In the language of the Command Pattern, the
 *  sampler is the Invoker or Command Manager.
 *
 *  After instantiating the sampler, users should add all the required steps using the Sampler::AddStep method, which places
 *  each step onto a stack. The entire sampling process is run using Sampler::Run.
 */
class Sampler {
public:
  /*! \brief  Constructor to initialize sampler.
   *
   *  \param options MCMC options object that defines sample_size, burnin, thinning interval, etc.
   */
  Sampler(MCMCOptions options) {
    cout << "Creating Sampler... " << endl;
    sample_size_ = options.sample_size;
    burnin_ = options.burnin;
    thin_ = options.thin;
    out_file_ = options.out_file;
  }
  /*! \brief Add Step to Sampler execution stack.
   *
   *  All parameters should have an associated Step that is added to the sampler.  The sampler stack
   *  defines one sampler iteration.
   *  \param step Step object for a given parameter.
   */
  void AddStep(Step* step) {
    // Add step to step stack.
    steps_.push_back(step);
    // Add index to tracking stack, if necessary tracked.
    if (steps_.back().ParameterTrack()) {
      tracks_.push_back(steps_.size() - 1);
    }
  }
  /*! \brief Run sampler for a specific number of iterations.
   *
   * \param number_of_iterations Number of iterations to run sampler.
   * \param progress Display a progress bar.
   */
	 void Iterate(int number_of_iterations, bool progress = false) {
	   if(progress) {
	     boost::progress_display show_progress(number_of_iterations);
	     for(int iter = 0; iter < number_of_iterations; ++iter) {
	       for(int i = 0; i < steps_.size(); ++i) {
	         steps_[i].DoStep();
	       }
	       ++show_progress;
	     }
	   }
	   else {
	     for(int iter = 0; iter < number_of_iterations; ++iter) {
	       for(int i = 0; i < steps_.size(); ++i) {
	         steps_[i].DoStep();
	       }
	     }
	   }
 	}

  /*! \brief Run sampler.
   *
   * Run the MCMC sampling procedure, including burning in, sampling, thinning, displaying progress,
   * and saving results.
   */
  void Run() {
    // Timer
    boost::timer timer;

    // Status of sampler...
    cout << "Running sampler..." << endl;
    cout << "Number of steps added: " << NumberOfSteps() << endl;
    cout << "Number of tracked steps added: " << NumberOfTrackedSteps() << endl;

    // Opening output file
    cout << "Opening output file: " << out_file_ << endl;
    ofstream outfile(out_file_.c_str());
    if (!outfile.is_open()) {
      cerr << "ERROR: Cannot open file!";
      exit(1);
    }

    for (int k = 0; k < tracks_.size(); ++k) {
      if (k == tracks_.size() - 1) {
        outfile << steps_[tracks_[k]].ParameterLabel() << endl;
      } else {
        outfile << steps_[tracks_[k]].ParameterLabel() << ",";
      }
    }

    // Setting starting value:
    cout << "Setting starting values..." << endl;
    for (int i = 0; i < steps_.size(); ++i) {
      steps_[i].Start();
    }

    // Burn in
    cout << "Burning in... (" << burnin_ << " iterations)" << endl;
    Iterate(burnin_, true);

    cout << endl << "Sampling..." << endl;
    boost::progress_display show_progress(sample_size_);
    for (int i = 0; i < sample_size_; ++i) {
      Iterate(thin_);
      // Save data
      for (int k = 0; k < tracks_.size(); ++k) {
        if (k == tracks_.size() - 1) {
          outfile << steps_[tracks_[k]].ParameterValue() << endl;
        } else {
          outfile << steps_[tracks_[k]].ParameterValue() << ",";
        }
      }
      //Show progress
      ++show_progress;
    }

    outfile.close();
		
		// Save lecuyer seed for future use
		Matrix<unsigned long, Col> seedmat(6, 1, false);
		unsigned long seed[6];
		myrng.GetState(seed);
		cout << "Saving seed: " << endl << seed[0] << " " << seed[1] << " " << seed[2] 
			<< " " << seed[3] << " " << seed[4] << " " << seed[5] << endl;
		for(int i = 0; i < 6; ++i) {
			seedmat(i) = seed[i];
		}
		seedmat.save("lecuyer.seed", 'o');
		
    cout << "Total elapsed time: " << timer.elapsed() << " seconds" << endl;
  }
  /*! \brief Number of steps in one sampler iteration.
   *
   * \return Number of steps.
   */
  int NumberOfSteps() {
    return steps_.size();
  }
  /*! \brief Number of tracked steps in one sampler iteration.
   *
   * \return Number of tracked steps.
   */
  int NumberOfTrackedSteps() {
    return tracks_.size();
  }
private:
  int sample_size_;
  int burnin_;
  int thin_;
  string out_file_;
  boost::ptr_vector<Step> steps_;
  vector<int> tracks_;
};

/*! \brief Approximately equal.
 *
 *  Checks if two doubles are within the numeric_limits<double>::epsilon() precision of each other.
 *  Useful if === is too strict.
 */
bool approx_equal(double a, double b) {
  return abs(a - b) <= numeric_limits<double>::epsilon() ? true : false;
}

/// Fast sum of log of dnorms
template <typename T, matrix_order PO, matrix_style PS>
double sum_fast_log_dnorm(const Matrix<T, PO, PS> &A, const double mean, const double sd) {
  double result = 0;

  for (int i = 0; i < A.size(); ++i) {
    // Slower part here is A[i], can this be speed up using an iterator?
    result += fast_log_dnorm(A[i], mean, sd);
  }
  return result;
}

/// Fast log of dnorm for mean parameter
double fast_log_dnorm(const double x, const double mean, const double sd) {
  return -pow((x - mean), 2) / (2*pow(sd, 2));
}



