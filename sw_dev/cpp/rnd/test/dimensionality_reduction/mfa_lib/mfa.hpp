// EM for Mixtures of Factor Analyzers
// Copyright (C) 2005  Kurt Tadayuki Miller and Mark Andrew Paskin
// 
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

#ifndef MFA_HPP
#define MFA_HPP

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_permute.h>
#include <gsl/gsl_permute_vector.h>

#include <string>
#include <stdexcept>
#include <vector>

#include <boost/random/mersenne_twister.hpp>

#include "log.hpp"

/**
 * An abstract interface to a collection of data vectors.  These
 * vectors are represented using GSL (GNU Scientific Library)
 * matrices, and they represent missing values using not-a-number
 * (NaN).  This interface permits repeated sequential scans of the
 * data set using a cursor interface.
 */
class vector_data_set_t {

public:
  //! Destructor.
  virtual ~vector_data_set_t() { };
    
  //! Returns the length of the data vectors.
  virtual std::size_t length() const = 0;

  //! Initializes the cursor to the start of the data set.
  virtual void reset() = 0;

  /**
   * Gets the next data vector in the data set, if there are data
   * vectors that have not yet been returned.
   *
   * @param  vector_ptr
   *         A GSL matrix with at least #length() rows and one 
   *         column.  If there is a next data vector to return,
   *         these elements are updated to represent the data
   *         vector.  The value not-a-number (NaN) is used
   *         to represent missing values.
   * @return true if there was a next data vector to return; false
   *         if the entire data set has been scanned
   */
  virtual bool get_next_vector(gsl_vector* vector_ptr) = 0;

}; // vector_data_set_t

/**
 * A mixture of factor analyzers (MFA).  This is a probabilistic
 * latent-variable model of vector-valued data \f$Y = [Y_1, Y_2,
 * \ldots, Y_n]\f$.  It posits two latent variables: \f$Z\f$, a
 * discrete variable with \f$k\f$ values, and a continuous vector
 * \f$X = [X_1, X_2, \ldots, X_m]\f$.  \f$Z\f$ is multinomial
 * distributed with parameters \f$\pi_1, \ldots, \pi_k\f$.  Each
 * \f$X_i\f$ is Gaussian distributed with zero mean and unit
 * variance.  The observed variable \f$Y\f$ is distributed according
 * to
 *
 * \f[ 
 *   Y \, | \, X = x, Z = i \sim N(\mu_i + W_i x, \Sigma)
 * \f]
 *
 * where \f$\Sigma = \textrm{diag}[(\sigma_1)^2, (\sigma_2)^2,
 * \ldots, (\sigma_n)^2)]\f$.
 *
 * This model includes two special cases.  If the number of mixture
 * components is set to one (\f$k = 1\f$), then the model is no
 * longer a mixture, but a simple factor analyzer.  If the
 * conditional variances of the \f$Y_i\f$ are forced to be equal,
 * i.e., \f$(\sigma_1)^2 = (\sigma_2)^2 = \ldots = (\sigma_n)^2 =
 * \sigma^2\f$, then the model is called a probabilistic PCA model.
 *
 * @see "The EM algorithm for mixtures of factor analyzers" by
 *      Zoubin Ghahramani and Geoff Hinton. Technical Report CRG-TR-96-1,
 *      University of Toronto.
 * @see "Probabilistic Principal Component Analysis" by Michael Tipping 
 *      and Christopher Bishop, Journal of the Royal Statistical Society, 
 *      Series B, 61, Part 3, pp. 611-622.
 */
class mfa_t {

public:

  /**
   * Constructor.  The MFA model is loaded from file.  This
   * constructor may throw a std::runtime_error exception if the
   * file cannot be opened, or if it is improperly formatted.
   *
   * @param path a path to the file containing the serialized model
   */
  mfa_t(std::string path) throw (std::runtime_error);
    
  /**
   * Constructor.  The parameters are initialized so that the factor
   * loading matrix \f$W\f$ is all ones, the prior means \f$\mu\f$
   * are zero, and the conditional variance(s) \f$\sigma^2\f$ are
   * one, and the mixture components \f$\pi\f$ are random.
   *
   * @param k    the number of mixture components, i.e., the arity of
   *             the latent discrete variable \f$Z\f$
   * @param m    the dimensionality of the latent continuous variable 
   *             \f$X\f$
   * @param n    the dimensionality of the observed continuous variable 
   *             \f$Y\f$
   * @param ppca A flag which, if true, enforces that 
   *             \f$\textrm{Cov}(Y \, | \, X = x, Z = z) = \sigma^2 I\f$
   *             rather than \f$\textrm{Cov}(Y \, | \, X = x, Z = z) = 
   *             \textrm{diag}[(\sigma_1)^2, (\sigma_2)^2, \ldots, 
   *             (\sigma_n)^2]\f$.  This type of model is called a 
   *             probabilistic PCA model (PPCA).
   */
  mfa_t(std::size_t k, std::size_t m, std::size_t n, bool ppca = false);

  //! Destructor
  virtual ~mfa_t();

  /**
   * Returns the number of mixture components, i.e., the arity of
   * the latent discrete variable \f$Z\f$.
   */
  std::size_t get_k() const { return k; }

  /**
   * Returns the dimensionality of the latent continuous variable
   * \f$X\f$.
   */
  std::size_t get_m() const { return m; }

  /**
   * Returns the dimensionality of the observed continuous variable
   * \f$Y\f$.
   */
  std::size_t get_n() const { return n; }

  /**
   * Returns true if \f$\textrm{Cov}(Y \, | \, X = x) = \sigma^2 I\f$
   * rather than \f$\textrm{Cov}(Y \, | \, X = x) = \textrm{diag}
   * [(\sigma_1)^2, (\sigma_2)^2, \ldots, (\sigma_n)^2]\f$.  This type of
   * model is called a probabilistic PCA model (PPCA).
   */
  bool is_ppca() const { return ppca; }

  /**
   * Resets the log likelihood used for training
   */
  void reset_log_likelihood();
    
  /**
   * Computes the log likelihood of a data vector under this mixture
   * of factor analyzers.
   * 
   * @param  data 
   *         An \f$n \times 1\f$ vector representing an observation
   *         of (a subset of) \f$Y\f$.  Elements that are 
   *         not-a-number (NaN) represent missing values.
   * @return The log likelihood of the observed values under this
   *         model (using the natural logarithm).
   */
  double log_likelihood(const gsl_vector* data);

  /**
   * Runs expectation-maximization (EM), an iterative algorithm to
   * train the parameters of this model from a data set, starting
   * from the current parameters.
   *
   * @param  data
   *         the data set from which the model is trained
   * @param  tol
   *         A convergence tolerance.  If in a single iteration the 
   *         log likelihood of the data increases by less than this 
   *         amount, the algorithm terminates.
   * @param  max_iter
   *         A limit on the number of iterations.
   * @return true if EM converged, and false if max_iter iterations
   *         were performed without convergence
   */
  bool em(vector_data_set_t& data,
	  double tol = 1e-1,
	  std::size_t max_iter = std::numeric_limits<std::size_t>::max());

  /**
   * This gets the parameters for ppca in a closed form solution as
   * described in Tipping and Bishop.  This does not store all the
   * data in memory and is therefore slower than #ppca_solve_fast.
   * This cannot be used when there is hidden data.
   * 
   * @param  data
   *         the data set from which the model is trained
   */
  void ppca_solve(vector_data_set_t& data);

  /**
   * This gets the parameters for ppca in a closed form solution as
   * described in Tipping and Bishop.  As opposed to the above, it
   * stores all data in memory.  This should only be tried if all
   * data can easily fit in memory or if it is costly to read in the
   * data.  This cannot be used when there is hidden data.
   * 
   * @param  data
   *         the data set from which the model is trained
   */
  void ppca_solve_fast(vector_data_set_t& data);

  /**
   * Saves this model to a file.  This method throws a
   * std::runtime_error exception if the file cannot be opened for
   * writing, or if an I/O error occurs.  The format of the file is
   * \f$n m k ppca \pi \Sigma \{\mu_i W_i\} \times k\f$
   *
   * @param path a path to the file where this model should be saved
   *
   * @return Whether or not this save was successful
   */
  bool save(const std::string& path) const throw(std::runtime_error);

  /**
   * Loads the parameters of the MFA from file.  This method may
   * throw an exception if an error occurs. The format of the file
   * is \f$n m k ppca \pi \Sigma \{\mu_i W_i\} \times k\f$
   *
   * @param path
   *        The name of the file to load the parameters from
   */
  virtual void load(const std::string& path) throw(std::runtime_error);

  /**
   * Converts a PPCA model into an FA model by letting
   * \f$\Sigma_{ii} = \Sigma\f$
   */
  void convert_PPCA_to_FA();
    
  /**
   * Converts an FA model into a PPCA model by letting
   * \f$\Sigma = \frac{1}{n}\sum \Sigma_{ii}\f$
   */
  void convert_FA_to_PPCA();

  /**
   * Gets the expected data vector
   * 
   * @param  data 
   *         An \f$n \times 1\f$ vector representing an observation
   *         of (a subset of) \f$Y\f$.  Elements that are 
   *         not-a-number (NaN) represent missing values.
   * @param  hidden_mask
   *         A mask of which data points are heuristically believed to be 
   *         forground.  
   * @param  expected_data
   *         An \f$n \times 1\f$ vector in which to store the
   *         expected data
   */
  void get_expected_data(const gsl_vector* data,
			 const std::vector<bool>& hidden_mask,
			 gsl_vector* expected_data);
    
  /**
   * Prints the \f$W\f$ for the desired factor out to to disk in
   * matlab ascii format
   *
   * @param  j 
   *         the index of the mixture component to print
   * @param filename
   *        The name of the file to print out to
   */
  void print_W_to_file(const std::size_t j,
		       const std::string& file_name) const;
    
  /**
   * Prints the \f$\mu\f$ for the desired factor out to to disk in
   * matlab ascii format
   *
   * @param  j 
   *         the index of the mixture component to print
   * @param filename
   *        The name of the file to print out to
   */
  void print_mu_to_file(const std::size_t j,
			const std::string& file_name) const;

  /**
   * Prints the \f$\Sigma\f$ out to to disk in matlab ascii format
   *
   * @param filename
   *        The name of the file to print out to
   */
  void print_sigma_to_file(const std::string& file_name) const;

protected:
  /**
   * Initializes the memory used by this module
   *
   * @param k    the number of mixture components, i.e., the arity of
   *             the latent discrete variable \f$Z\f$
   * @param m    the dimensionality of the latent continuous variable 
   *             \f$X\f$
   * @param n    the dimensionality of the observed continuous variable 
   *             \f$Y\f$
   * @param ppca A flag which, if true, enforces that 
   *             \f$\textrm{Cov}(Y \, | \, X = x, Z = z) = \sigma^2 I\f$
   *             rather than \f$\textrm{Cov}(Y \, | \, X = x, Z = z) = 
   *             \textrm{diag}[(\sigma_1)^2, (\sigma_2)^2, \ldots, 
   *             (\sigma_n)^2]\f$.  This type of model is called a 
   *             probabilistic PCA model (PPCA).
   */
  void initialize(std::size_t k, std::size_t m, 
		  std::size_t n, bool ppca = false);

  /** 
   * Performs the SVD decomposition needed for the PPCA closed form
   * solution and stores the results in #Sigma and the corresponding
   * \f$\mu\f$ and \f$W\f$. Note that the SVD operation changes the
   * data stored in at_a_matrix.
   *
   * @param at_a_matrix
   *        Stores the result of \f$A^\top A\f$
   * @param data
   *        the data set from which the model is trained
   * @param num_data_points
   *        The number of data points accumulated
   */
  void ppca_svd_solve(gsl_matrix* at_a_matrix, 
		      vector_data_set_t& data,
		      const std::size_t num_data_points);

  /**
   * Calcualates the proper permutation needed to separate the 
   * hidden data from the observed data.
   * 
   * @param permutation
   *        The permutation that will seperate the data
   * @param data
   *        The input data
   *
   * @return 
   *        The total number of hidden values
   */
  static std::size_t compute_permutation(gsl_permutation* permutation, 
					 const gsl_vector* data);

  /**
   * Calcualates the proper permutation needed to separate the 
   * hidden data from the observed data.
   * 
   * @param permutation
   *        The permutation that will seperate the data
   * @param hidden_mask
   *        The mask of which values are hidden
   *
   * @return 
   *        The total number of hidden values
   */
  static std::size_t compute_permutation(gsl_permutation* permutation, 
					 const std::vector<bool>& hidden_mask);

  /**
   * Permutates the paramters of one of the mixture components
   *
   * @param  j 
   *         the index of the mixture component to permute
   * @param permutation
   *         the permutation to perform on the parameters
   * @param num_hidden
   *         the number of hidden variables
   */
  void permute_parameters(const std::size_t j, 
			  const gsl_permutation* permutation);

  /**
   * Permutates the sufficient statistics
   *
   * @param permutation
   *        the permutation to perform on the sufficient statistics
   */
  void permute_sufficient_statistics(const gsl_permutation* permutation);

  /**
   * Computes the average value in the data set
   *
   * @param  data
   *         the data set to be used
   */
  void compute_mu(vector_data_set_t& data);

  /**
   * Computes \f$M\f$ for one of the mixture components using the
   * observation.  (See the documentation for #M.)  The factor
   * scratch members \f$M_j\f$, \f$M^{-1}\f$, and the LU
   * decomposition of \f$M\f$ are updated.
   *
   * @param  j 
   *         the index of the mixture component for which \f$M\f$
   *         is computed
   * @param  num_hidden
   *         The number of \f$y\f$ values that are hidden
   * @param  to_scratch
   *         A flag to indicate where to save the results.  If true, then
   *         The results are saved to the local scratch data, otherwise
   *         they are saved to the variables in the mfa class.
   */
  void compute_M(const std::size_t j, const std::size_t num_hidden,
		 const bool to_scratch = false) const;

  /**
   * Computes the value of \f$q\f$ for one of the mixture components
   * using the observation.  (See the documentation for #q.)  The
   * local member #q is updated to represent this value.
   *
   * @param  j 
   *         the index of the mixture component for which \f$q\f$ 
   *         is computed
   * @param  data 
   *         An \f$n \times 1\f$ vector representing an observation
   *         of (a subset of) \f$Y\f$.  Elements that are 
   *         not-a-number (NaN) represent missing values.
   * @param  num_hidden
   *         The number of \f$y\f$ values that are hidden
   * @param  to_scratch
   *         A flag to indicate where to save the results.  If true, then
   *         The results are saved to the local scratch data, otherwise
   *         they are saved to the variables in the mfa class.
   */
  void compute_q(const std::size_t j, const gsl_vector* data,
		 const std::size_t num_hidden,
		 const bool to_scratch = false) const;

  /**
   * Computes the sufficient statistics needed for factor analysis.  
   * This requires that #M_lu and #q have been pre-computed;
   *
   * \f{eqnarray*}
   * M &= &I + W_o^\top \Sigma^{-1} W_o\\
   * q &= &(y_o - \mu_o)^\top \Sigma^{-1}W_o\\
   * \langle x \rangle  &= &(I + W_o^\top \Sigma^{-1} W_o)^{-1} W_o^\top \Sigma^{-1}(y_o - \mu_o)\\
   * &=& M^{-1}q^\top \\
   * \langle y_h\rangle &= &W_h(I + W_o^\top \Sigma^{-1} W_o)^{-1} W_o^\top \Sigma^{-1}(y_o - \mu_o) + \mu_h\\
   * &= &W_h \langle x\rangle + \mu_h\\
   * \langle y_o \rangle &= &y_o\\
   * \langle xx^\top \rangle  &= & (I + W_o^\top \Sigma^{-1} W_o)^{-1} + \langle x\rangle \langle x\rangle^\top \\
   * &=& M^{-1} + \langle x\rangle \langle x\rangle^\top \\
   * \langle y_hx^\top \rangle  &= & W_h (I + W_o^\top \Sigma^{-1} W_o)^{-1} + \langle y_h\rangle \langle x\rangle^\top \\
   * &=&W_h M^{-1} + \langle y_h\rangle \langle x\rangle^\top \\
   * \langle y_ox^\top \rangle  &= & y_o\langle x \rangle^\top \\
   * \langle y_hy_h^\top \rangle  &= & \Sigma + W_h \langle xx^\top \rangle  W_h^\top  + 2 W_h \langle x\rangle  \mu_h^\top  + \mu_h \mu_h^\top \\
   * \langle y_hy_h^\top \rangle_{ii}  &= &\Sigma_{ii} + W_i\langle xx^\top \rangle W_i^\top  + 2 W_i \langle x\rangle \mu_i + \mu_i^2\\
   * \langle y_oy_o^\top \rangle  &= &y_oy_o^\top \\
   * \f}
   *
   * @param  j 
   *         the index of the mixture component for which the 
   *         log likelihood is computed
   * @param  data 
   *         An \f$n \times 1\f$ vector representing an observation
   *         of (a subset of) \f$Y\f$.  Elements that are 
   *         not-a-number (NaN) represent missing values.
   * @param  num_hidden
   *         The number of \f$y\f$ values that are hidden
   */
  void compute_sufficient_statistics(const std::size_t j,
				     const gsl_vector* data,
				     const std::size_t num_hidden);

  /**
   * Accumulates the sufficient statistics needed for factor analysis.  
   * This requires that the sufficient statistics have been pre-computed;
   *
   * \f{eqnarray*}
   * W_1 &=& \left[\begin{array}{cc} \sum_i^N  h_{ij} \left\langle yx^\top\right\rangle& \sum_i^N  h_{ij}\left\langle y\right\rangle \end{array} \right]\\
   * W_2 &=& \left[\begin{array}{cc} 
   * \sum_i^N  h_{ij} \left\langle xx^\top\right\rangle & \sum_i^N  h_{ij} \left\langle x\right\rangle\\
   * \sum_i^N  h_{ij} \left\langle x\right\rangle^\top  & \sum_i^N  h_{ij}\\
   * \end{array}\right]\\
   * \Sigma_1 &=& \sum_i^N  h_{ij} \textrm{diag}\left(\left\langle yy^\top\right\rangle\right)\\
   * h_1 &=& \sum_i^N  h_{ij}\\
   * \f}
   * 
   * @param  j 
   *         the index of the mixture component we are accumulating
   *         statistics for
   * @param  h
   *         the factor weight as described in Ghahramani and Hinton.
   */
  void accumulate_sufficient_statistics(const std::size_t j, const double h);

  /**
   * Recomputes the parameters for each of the factor analyzers.  Requires
   * that the accumulators have been pre-computed.  We do a joint estimation
   * of \f$\mu\f$ and \f$W\f$.
   *
   * \f{eqnarray*}
   * \left[\begin{array}{cc}\widetilde{W} & \widetilde{\mu}\end{array}\right] 
   *  &=& \left[\begin{array}{cc} \sum_i^N  h_{ij} \left\langle yx^\top\right\rangle& \sum_i^N  h_{ij}\left\langle y\right\rangle \end{array} \right]
   * \left[\begin{array}{cc} 
   * \sum_i^N  h_{ij} \left\langle xx^\top\right\rangle & \sum_i^N  h_{ij} \left\langle x\right\rangle\\
   * \sum_i^N  h_{ij} \left\langle x\right\rangle^\top  & \sum_i^N  h_{ij}\\
   * \end{array}\right]^{-1}\\
   * &=& W_1 W_2 ^{-1}\\
   * \widetilde{\pi_j} &=& \frac{1}{N}\sum_i^N  h_j\\
   * &=& \frac{1}{N} h_1\\
   * \f}
   *
   * @param  j 
   *         the index of the mixture component for which we are 
   *         recomputing the paramters
   * @param  N
   *         the number of data points that were accumulated
   * @param  eta
   *         the over-relaxatation parameter
   */
  void recompute_parameters(const std::size_t j, const std::size_t N, 
			    const double eta);

  /**
   * Recomputes the #sigma paramter
   *
   * \f{eqnarray*}
   * \widetilde{\Sigma} &=& 
   * \frac{1}{N}\sum_j^k \textrm{diag}\left\{ \sum_i^N h_{ij} \left\langle yy^\top\right\rangle - 
   * \sum_i^N h_{ij} \left\langle yx^\top\right\rangle \widetilde{W}^\top -
   * \sum_i^N h_{ij} \left\langle y\right\rangle \widetilde{\mu}^\top
   * \right\}\\
   * &=& \frac{1}{N} \sum_j^k \left\{ \Sigma_1 - 
   * \textrm{diag} 
   * \left[
   * W_1\left(\begin{array}{c}\widetilde{W}^\top \\ \widetilde{\mu}^\top\end{array}\right)\right]
   * \right\}\\
   * \f}
   * 
   * and in the case of PPCA
   * \f[
   * \sigma^2 = \frac{\mathrm{tr}(\widetilde{\Sigma})}{n}
   * \f]
   * 
   * @param  N
   *         the number of data points that were accumulated
   * @param  eta
   *         the over-relaxatation parameter
   */
  void recompute_sigma(const std::size_t N, const double eta);

  /**
   * Resets the over-relaxation paramters
   */
  void reset_over_relaxation();

  /**
   * Computes the log likelihood of a data vector under one of the
   * mixture components.  This requires that #M_lu and #q have
   * been pre-computed.
   *
   * To compute the likelihood of the data vector, we make use of 
   * the following formula:
   * \f[
   *     p(y) = N(\mu, W W^\top + \Sigma)
   * \f]
   * To avoid computing and inverting this \f$n \times n\f$
   * covariance matrix, we use one the matrix inversion lemma.
   * This yields
   * \f[
   *   (W W^\top + \Sigma)^{-1} = 
   *   \Sigma^{-1} - \Sigma^{-1} W (I + W^\top \Sigma^{-1} W)^{-1}
   *      W^\top \Sigma^{-1}
   * \f]
   * Plugging this into the Gaussian model, we get that the 
   * likelihood is
   * \f[
   *   p(y) = \frac{1}{Z} \exp \left\{
   *    -\frac{1}{2} [(y - \mu)^\top \Sigma^{-1}(y - \mu) - q M^{-1}q^\top]
   *   \right\}
   * \f]
   *
   * where \f$q\f$ is defined as in the documentation for #q,
   * \f$M\f$ is defined as in the documentation for #M, and
   * \f$Z\f$ is the normalization constant which is \f$(2 \pi)^{-n/2}
   * |W W^\top + \Sigma|^{-1/2}\f$.  To compute this determinant, we
   * use the fact that \f$|W W^\top + \Sigma| = |\Sigma| \cdot |I +
   * W^\top \Sigma^{-1} W| = |\Sigma| \cdot |M|\f$.
   * 
   * @param  j 
   *         the index of the mixture component for which the 
   *         log likelihood is computed
   * @param  data 
   *         An \f$n \times 1\f$ vector representing an observation
   *         of (a subset of) \f$Y\f$.  Elements that are 
   *         not-a-number (NaN) represent missing values.
   * @return The log likelihood of the observed values under the 
   *         mixture component (using the natural logarithm).
   * @param  num_hidden
   *         The number of \f$y\f$ values that are hidden
   * @param  from_scratch
   *         A flag to indicate where to get \f$M\f$ and \f$q\f$ from. 
   *         If true, then they are from the local scratch data, otherwise
   *         they are from the local variables in the mfa class.
   */
  double log_likelihood(const std::size_t j, const gsl_vector* data,
			const std::size_t num_hidden,
			const bool from_scratch = false) const;

  /** 
   * Returns a random number sampled from the normal distribution
   * \f${\cal N}(0, \sigma)\f$
   *
   * @param sigma  
   *        The standard deviation of the process
   *
   * @return 
   *        A number drawn randomly from \f${\cal N}(0, \sigma)\f$
   */
  double sample_normal(const double sigma);


  /**
   * Draws a random sample from (0,1)
   *
   * @return
   *       A random number between 0 and 1
   */
  double sample_uniform01();

protected:
  /**
   * Information about one factor analyzer model in the mixture.
   */
  struct fa_info_t {
    /**
     * Constructor.
     *
     * @param m the dimensionality of the latent continuous 
     *          variable \f$X\f$
     * @param n the dimensionality of the observed continuous 
     *          variable \f$Y\f$
     */
    fa_info_t(std::size_t m, std::size_t n);

    //! Destructor.
    ~fa_info_t();

    //! The \f$n \times m\f$ factor loading matrix \f$W\f$.
    gsl_matrix* W;

    //! The \f$n \times 1\f$ prior mean vector \f$\mu\f$.
    gsl_vector* mu;
    
  };

  /**
   * Scratch space for data for data that only needs one copy
   */
  struct mfa_scratch_t {
    /**
     * Constructor.
     *
     * @param m the dimensionality of the latent continuous 
     *          variable \f$X\f$
     * @param n the dimensionality of the observed continuous 
     *          variable \f$Y\f$
     * @param ppca a flag indicating whether this is performing
     *          PPCA or FA.
     *         
     */
    mfa_scratch_t(std::size_t m, std::size_t n, bool ppca);

    //! Destructor.
    ~mfa_scratch_t();

    /**
     * An \f$n \times 1\f$ vector to store the  expected 
     * \f$\langle y \rangle\f$ value
     */
    gsl_vector* expected_y;
      
    /**
     * An \f$n \times m\f$ matrix to store the expected 
     * \f$\langle yx^\top  \rangle\f$ value
     */
    gsl_matrix* expected_yx;

    /**
     * An \f$n \times 1\f$ vector to store the expected diagonal of the 
     * \f$ \langle yy^\top \rangle\f$ matrix
     */
    gsl_vector* expected_yy;

    /**
     * The em prediction of the \f$\Sigma\f$ used by over-relaxation
     */
    gsl_vector* sigma_em;

    /**
     * A permutation for an \f$m+1 \times m\f$ matrix.
     */
    gsl_permutation* m1_permutation;
      
    /**
     * An \f$m+1 \times m+1\f$ scratch vector.
     */
    gsl_matrix* m1_by_m1_scratch;

    /**
     * An \f$n \times m+1\f$ scratch vector.
     */
    gsl_matrix* n_by_m1_scratch;
  };


  /**
   * Scratch space for data that is needed on a per factor basis.
   * Includes an accumulator of sufficient statistics needed to
   * recompute paramters for factor analysis and other per-factor
   * information needed in the training.
   */ 
  struct fa_factor_scratch_t {
    /**
     * Constructor.
     *
     * @param m the dimensionality of the latent continuous 
     *          variable \f$X\f$
     * @param n the dimensionality of the observed continuous 
     *          variable \f$Y\f$
     */
    fa_factor_scratch_t(std::size_t m, std::size_t n);

    //! Destructor.
    ~fa_factor_scratch_t();

    /**
     * An \f$n \times m\f$ matrix to store the accumulator for \f$W\f$
     *
     * \f$W_1 = \sum(\langle yx^\top \rangle  - 
     * \mu\langle x\rangle^\top )\f$
     */
    gsl_matrix* W1;

    /**
     * An \f$m \times m\f$ matrix to store the accumulator for \f$W\f$
     *
     * \f$W_2 = \sum(\langle xx^\top \rangle )\f$
     */
    gsl_matrix* W2;

    /**
     * An \f$n \times 1\f$ accumulator for \f$sigma\f$
     */
    gsl_vector* sigma1;

    /**
     * An accumulator for \f$h\f$ as described in Ghahramani and Hinton
     */
    double h1;

    /**
     * The weight times the likelihood for the current datum,
     * represented in log-space.
     */
    prl::log_t<double> posterior;

    /**
     * The current M matrix for the current factor
     */
    gsl_matrix* M_j;

    /**
     * The current LU decomposition of the M matrix for the current factor
     */
    gsl_matrix* M_j_lu;

    /**
     * The current inverse M matrix for the current factor
     */
    gsl_matrix* M_j_inv;

    /**
     * The sign used in computing the LU decomposition
     */
    int sign;
      
    /**
     * The permutation used for the matrix inversion
     */
    gsl_permutation* m_permutation;

    /**
     * The q vector for the current factor
     */
    gsl_vector* q_j;

    /**
     * The em prediction of the \f$W\f$ used by over-relaxation
     */
    gsl_matrix* W_em;

    /**
     * The em prediction of the \f$\mu\f$ used by over-relaxation
     */
    gsl_vector* mu_em;
  };
    
protected:

  /**
   * The number of mixture components, i.e., the arity of the latent
   * discrete variable \f$Z\f$.
   */
  std::size_t k;

  //! The dimensionality of the latent continuous variable \f$X\f$.
  std::size_t m;

  //! The dimensionality of the observed continuous variable \f$Y\f$.
  std::size_t n;

  /**
   * A flag which, if true, enforces that \f$\textrm{Cov}(Y \, | \, X
   * = x, Z = z) = \sigma^2 I\f$ rather than 
   * \f$\textrm{Cov}(Y \, | \, X = x, Z = z) = 
   * \textrm{diag}[(\sigma_1)^2, (\sigma_2)^2, \ldots,
   * (\sigma_n)^2]\f$.  This type of model is called a probabilistic
   * PCA model (PPCA).
   */
  bool ppca;

  /**
   * A vector of \f$k\f$ values representing the mixing proportions
   * of the model.  \f$\pi_i\f$ is the prior probability a sample is
   * generated from the \f$i^{\textrm{th}}\f$ factor analyzer.
   */
  gsl_vector* pi;

  /**
   * A vector of objects representing the components of the mixture
   * model.
   */
  std::vector<fa_info_t*> fa_ptr_vec;

  /**
   * A vector of objects
   */
  mutable std::vector<fa_factor_scratch_t*> fa_factor_scratch_vec;
    
  /**
   * A vector which represents the conditional covariance of \f$Y\f$
   * given \f$X\f$ implicitly.  If this model is a PPCA model, then
   * this vector is \f$1 \times 1\f$, and its sole entry is
   * \f$\sigma^2\f$.  Otherwise, this is an \f$n \times 1\f$ vector,
   * and its elements are \f$(\sigma_1)^2, (\sigma_2)^2, \ldots,
   * (\sigma_n)^2\f$.
   */
  gsl_vector* sigma;

  /**
   * The inverse of #sigma
   */
  gsl_vector* sigma_inv;

  /**
   * An \f$m \times m\f$ temporary storage matrix.  Fix one of the
   * mixture components, and partition \f$Y\f$ into \f$Y_o\f$ and
   * \f$Y_h\f$, corresponding to which values are observed and which
   * are hidden (in some data vector).  This field is used to store
   * the symmetric matrix \f$M\f$ where
   * \f[
   *     M = I_{m \times m} + W_o^\top \Sigma_{oo}^{-1} W_o
   * \f] 
   * where \f$W_o\f$ contains the rows of \f$W\f$ that correspond to
   * \f$Y_o\f$ and \f$\Sigma_{oo}\f$ is the subblock of \f$\Sigma\f$
   * corresponding to the observed components.  The inverse of this
   * matrix and its determinant appear in the expression for
   * computing the likelihood of data \f$p(Y_o = y_o)\f$ and the
   * expression for computing the posterior expectations \f$E[X \, |
   * \, Y_o = y_o]\f$, \f$E[XX^\top \, | \, Y_o = y_o]\f$, and
   * \f$E[Y_h X^\top \, | \, Y_o = y_o]\f$.  
   *
   * \f$M\f$ is the posterior inverse covariance (or information
   * matrix) of \f$X\f$ given some set of observations.
   */
  mutable gsl_matrix* M;

  /**
   * An \f$m \times m\f$ temporary storage matrix used to store the
   * LU decomposition of \f$M\f$ (see #M).  This matrix is
   * used to compute the inverse and the determinant of \f$M\f$.
   */
  mutable gsl_matrix* M_lu;

  /**
   * An \f$m \times m\f$ temporary storage matrix.  This stores the
   * inverse of the \f$M\f$ matrix (See #M).
   */
  mutable gsl_matrix* M_inv;

  /**
   * A permutation for an \f$m \times m\f$ matrix.  Used in the calculation
   * of #M_inv.
   */
  mutable gsl_permutation* m_permutation;

  /**
   * A permutation for an \f$n \times m\f$ matrix.  
   */ 
  mutable gsl_permutation* n_permutation;

  /**
   * A permutation for an \f$n \times m\f$ matrix.  This is the inverse
   * of the #n_permutation
   */ 
  mutable gsl_permutation* n_permutation_inv;
    
  /**
   * The sign used by the m_permutation
   */
  mutable int sign;

  /**
   * An \f$m \times 1\f$ temporary storage vector.  Fix one of the
   * mixture components, and partition \f$Y\f$ into \f$Y_o\f$ and
   * \f$Y_h\f$, corresponding to which values are observed and which
   * are hidden (in some data vector).  This vector is used to store
   * the value
   *
   * \f[
   *     q = (y_o - \mu_o)^\top \Sigma_{oo}^{-1} W_o
   * \f] 
   *
   * where \f$y_o\f$ is the observation, \f$\mu_o\f$ are the prior
   * means of the observed components of \f$Y\f$,
   * \f$\Sigma_{oo}\f$ is the subblock of \f$\Sigma\f$
   * corresponding to the observed components, and \f$W_o\f$ is
   * the rows of the factor loading matrix corresponding to the
   * observed components.  This vector appears in the expression
   * for computing the likelihood of data \f$p(Y_o = y_o)\f$ and
   * the expression for computing the posterior expectation
   * \f$E[X \, | \, Y_o = y_o]\f$.
   *
   * \f$q\f$ is the posterior information vector of \f$X\f$ given
   * some set of observations.
   */
  mutable gsl_vector* q;

  /**
   * An \f$m \times 1\f$ vector to store the expected <x> value
   */
  mutable gsl_vector* expected_x;
    
  /**
   * An \f$m \times m\f$ matrix to store the expected <xx'> value
   */
  mutable gsl_matrix* expected_xx;

  //! An \f$m \times 1\f$ scratch vector.
  mutable gsl_vector* m_by_1_scratch;

  //! An \f$n \times 1\f$ scratch vector.
  mutable gsl_vector* n_by_1_scratch;

  //! An \f$m \times m\f$ scratch vector.
  mutable gsl_matrix* m_by_m_scratch;

  //! An \f$n \times m\f$ scratch vector.
  mutable gsl_matrix* n_by_m_scratch;

  /**
   * An \f$m \times 1\f$ vector to store the current data from the 
   * vector_data_set_t
   */
  mutable gsl_vector* current_data;

  /**
   * The bank of scratch memory used to do all calculations
   */
  mutable mfa_scratch_t* mfa_scratch_ptr;

  /**
   * A flag representing whether or not the memory of this class has
   * been allocated
   */
  bool initialized;

  /**
   * A flag representing whether or not the \f$\mu\f$ of the factors
   * have been initialized
   */ 
  bool mu_initialized;

  //! The most recent log likelihood computation
  double previous_log_likelihood;

  //! The eta paramter used by over-relaxation
  double eta;

  /**
   * A pseudo-random number generator used to generate noise in the
   * initialization 
   */
  boost::mt19937 prng;

}; // class mfa_t

#endif // #ifndef MFA_HPP

