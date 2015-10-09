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

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <cmath>
#include <iostream>

#include <boost/random.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
//--S [] 2015/08/05 : Sang-Wook Lee
#if defined(_MSC_VER)
#include <boost/math/special_functions/fpclassify.hpp>
using namespace boost::math;  // for isnan().
#endif
//--E [] 2015/08/05 : Sang-Wook Lee

#include "mfa.hpp"


/**
 * (This should later on go in mfa.hpp)
 * Todos:
 *       thoroughly test MFA
 *       figure out why MFA can have a mean outside of values contained
 *              in the data
 *
 * --------------------------------------
 * PPCA
 * --------------------------------------
 * 
 * PPCA has been thoroughly tested and can be trained in 1 of 3 ways:
 * EM, "solve," and "solve_fast."
 * 
 * EM finds the solution in an iterative manner and is slow.  However,
 * it is the only one that supports missing data points, so if there
 * are missing/hidden data points, this is the only method to solve
 * the problem. 
 *
 * "solve_fast" reads all data into memory and solves the closed form
 * solution.  If all the data fits in memory, this is the prefered
 * way.
 * 
 * If the data does not fit in memory, "solve" still does the closed
 * form solution, but has to go through the data O(N) times and can
 * therefore be significantly slower.
 * 
 * -------------------------------------
 * FA
 * --------------------------------------
 * 
 * FA has been decently tested.  If running MFA with one component, it
 * is recommended to first run ppca_solve or ppca_solve_fast to
 * initialize with a good first estimate.  At that point,
 * convert_PPCA_to_FA can be used to switch to FA and then use em will
 * solve the rest.
 * 
 * -------------------------------------
 * MFA
 * --------------------------------------
 * 
 * MFA has only been tested on small data sets.  With a larger data
 * set, it gave unexpected results, but the cause has not been
 * determined.
 * 
 * -------------------------------------
 * Training
 * --------------------------------------
 *
 * To train data with this class, use mfa_train and add a wrapper
 * class that will read in the data and put it in GSL format.  An
 * example is included in mfa_train.
 *
 * -------------------------------------
 * Testing
 * --------------------------------------
 * 
 * To use this as a classifier, use one of the likelihood functions.
 * To see what the expected data would be, use get_expected_data.
 * This is helpful in testing proper convergence/functionality.
 * 
 */

//==================== debug utils =========================
void matrix_printf(const gsl_matrix* m, const bool one_line) {
  for (unsigned int r = 0; r < m->size1; ++r) {
    for (unsigned int c = 0; c < m->size2; ++c)
      printf("%.5lf ", gsl_matrix_get(m, r, c));
    if (!one_line) 
      printf("\n");
  }
  if (one_line)
    printf("\n");
}

void vector_printf(const gsl_vector *v) {
  for (unsigned int i = 0; i < v->size; ++i)
    printf("%.5lf ", gsl_vector_get(v, i));
  printf("\n");
}

//======================== mfa_t::fa_info_t ==============================
mfa_t::fa_info_t::fa_info_t(std::size_t m, std::size_t n) {
  //allocate memory
  this->W = gsl_matrix_alloc(n, m);
  this->mu = gsl_vector_alloc(n);
}

mfa_t::fa_info_t::~fa_info_t() {
  //free memory
  gsl_matrix_free(this->W);
  gsl_vector_free(this->mu);
}

//======================== mfa_t::mfa_scratch_t ==============================
mfa_t::mfa_scratch_t::mfa_scratch_t(std::size_t m, std::size_t n, 
                                           bool ppca) {
  //allocate memory
  this->expected_yx = gsl_matrix_alloc(n, m);
  this->expected_y  = gsl_vector_alloc(n);
  this->expected_yy = gsl_vector_alloc(n);
  if (ppca)
    this->sigma_em = gsl_vector_alloc(1);
  else 
    this->sigma_em = gsl_vector_alloc(n);
  this->m1_by_m1_scratch = gsl_matrix_alloc(m+1, m+1);
  this->n_by_m1_scratch = gsl_matrix_alloc(n, m+1);
  this->m1_permutation = gsl_permutation_alloc(m+1);
}

mfa_t::mfa_scratch_t::~mfa_scratch_t() {
  //free memory
  gsl_matrix_free(this->expected_yx);
  gsl_vector_free(this->expected_y);
  gsl_vector_free(this->expected_yy);
  gsl_vector_free(this->sigma_em);
  gsl_matrix_free(this->m1_by_m1_scratch);
  gsl_matrix_free(this->n_by_m1_scratch);
  gsl_permutation_free(this->m1_permutation);
}

//====================== mfa_t::fa_factor_scratch_t ==========================
mfa_t::fa_factor_scratch_t::fa_factor_scratch_t(std::size_t m,
                                                       std::size_t n) {
  //allocate memory
  this->W1     = gsl_matrix_alloc(n, m+1);
  this->W2     = gsl_matrix_alloc(m+1, m+1);
  this->W_em   = gsl_matrix_alloc(n, m);
  this->sigma1 = gsl_vector_alloc(n);
  this->mu_em  = gsl_vector_alloc(n);
  this->M_j     = gsl_matrix_alloc(m, m);
  this->M_j_lu  = gsl_matrix_alloc(m, m);
  this->M_j_inv = gsl_matrix_alloc(m, m);
  this->q_j     = gsl_vector_alloc(m);
  this->m_permutation = gsl_permutation_alloc(m);
}

mfa_t::fa_factor_scratch_t::~fa_factor_scratch_t() {
  //free memory
  gsl_matrix_free(this->W1);
  gsl_matrix_free(this->W2);
  gsl_matrix_free(this->W_em);
  gsl_vector_free(this->sigma1);
  gsl_vector_free(this->mu_em);
  gsl_matrix_free(this->M_j);
  gsl_matrix_free(this->M_j_inv);
  gsl_matrix_free(this->M_j_lu);
  gsl_vector_free(this->q_j);
  gsl_permutation_free(this->m_permutation);
}

//======================== mfa_t public ==================================
mfa_t::mfa_t(std::string path) throw (std::runtime_error) : 
  mfa_scratch_ptr(NULL), initialized(false), mu_initialized(false) {
  //load the data, which in turn initializes the model
  load(path);
}

mfa_t::mfa_t(std::size_t k, std::size_t m, std::size_t n, bool ppca) :
  k(k), m(m), n(n), ppca(ppca), mfa_scratch_ptr(NULL), initialized(false),
  mu_initialized(false) {
  //initialize the model
  initialize(k, m, n, ppca);
}

mfa_t::~mfa_t() {
  // free any allocated memory
  if (initialized) {
    gsl_matrix_free(this->expected_xx);
    gsl_matrix_free(this->M);
    gsl_matrix_free(this->M_lu);
    gsl_matrix_free(this->M_inv);
    gsl_matrix_free(this->m_by_m_scratch);
    gsl_matrix_free(this->n_by_m_scratch);
    gsl_vector_free(this->expected_x);
    gsl_vector_free(this->q);
    gsl_vector_free(this->m_by_1_scratch);
    gsl_vector_free(this->n_by_1_scratch);
    gsl_vector_free(this->current_data);
    gsl_vector_free(this->sigma);
    gsl_vector_free(this->sigma_inv);
    gsl_permutation_free(this->m_permutation);
    gsl_permutation_free(this->n_permutation);
    gsl_permutation_free(this->n_permutation_inv);
    gsl_vector_free(this->pi);
  }
}

void mfa_t::reset_log_likelihood() {
  previous_log_likelihood = -std::numeric_limits<double>::infinity();
}

double mfa_t::log_likelihood(const gsl_vector* data) {
  //permute the data
  gsl_blas_dcopy(data, current_data);
  std::size_t num_hidden = compute_permutation(n_permutation, current_data);
  gsl_permutation_inverse(n_permutation_inv, n_permutation);
  gsl_permute_vector(n_permutation, current_data);
  if (!ppca) {
    gsl_permute_vector(n_permutation, sigma);
    gsl_permute_vector(n_permutation, sigma_inv);
  }
  //add the log likelihood for each factor
  prl::log_t<double> ll;
  for (std::size_t j = 0; j < k; ++j) {
    permute_parameters(j, n_permutation);
    this->compute_M(j, num_hidden); 
    this->compute_q(j, current_data, num_hidden); 
    prl::log_t<double> log_prior(gsl_vector_get(pi, j));
    prl::log_t<double> 
      log_lh(this->log_likelihood(j, current_data, num_hidden),
             prl::log_tag_t());
    ll += log_prior * log_lh;
    permute_parameters(j, n_permutation_inv);
  }
  //unpermute the sigmas
  if (!ppca) {
    gsl_permute_vector(n_permutation_inv, sigma);
    gsl_permute_vector(n_permutation_inv, sigma_inv);
  }
  return ll.get_log_value();
}

bool mfa_t::em(vector_data_set_t& data, double tol,
                      std::size_t max_iter) {
  assert(initialized);
  bool converged = false;
  // alpha is the over-relaxation step
  double alpha = 1.1;
  //Note: we will only perform over-relaxation when we have 1 component
  if (k > 1)
    alpha = 1.0;
  //reset the average images if we need to
  if (!mu_initialized) {
    printf("Computing the average data value... ");
    fflush(stdout);
    compute_mu(data);
    for (std::size_t j = 0; j < k; ++j)
      gsl_blas_dcopy(current_data, fa_ptr_vec[j]->mu);
    printf("done!\n      ");
  }
  // If this is the first time around, initialize all memory we need
  if (mfa_scratch_ptr == NULL) {
    mfa_scratch_ptr = new mfa_t::mfa_scratch_t(this->m, this->n, this->ppca);
    for (std::size_t j = 0; j < k; ++j) {
      fa_factor_scratch_vec[j] =
        new mfa_t::fa_factor_scratch_t(this->m, this->n);
      gsl_blas_dcopy(fa_ptr_vec[j]->mu, fa_factor_scratch_vec[j]->mu_em);
      gsl_matrix_memcpy(fa_factor_scratch_vec[j]->W_em, fa_ptr_vec[j]->W);
    }
    gsl_blas_dcopy(this->sigma, mfa_scratch_ptr->sigma_em);
    //and set the over-relaxation parameter
    eta = 1.0;
  }
  // repeat until convergence / num repetitions 
  std::size_t iteration = 0;
  while(iteration < max_iter) {
    iteration++;
    double current_log_likelihood = 0.0;
    std::size_t N = 0;
    // clear all accumulators
    for (std::size_t j = 0; j < k; ++j) {
      fa_factor_scratch_t* fa_factor_scratch = fa_factor_scratch_vec[j];
      gsl_matrix_set_zero(fa_factor_scratch->W1);
      gsl_matrix_set_zero(fa_factor_scratch->W2);
      gsl_vector_set_zero(fa_factor_scratch->sigma1);
      fa_factor_scratch->h1 = 0;
    }
    // go through all the data
    data.reset();
    while(true) {
      if (data.get_next_vector(current_data) == false)
        break;
      //compute the permutation and number of hidden data points
      std::size_t num_hidden = 
        compute_permutation(n_permutation, current_data);
      if (num_hidden == n)
        continue;  //ignore any completely empty frames
      N++;
      if (N % 10 == 0) { // visual feedback of the progress
        printf("."); fflush(stdout);
      }
      // permute the data
      gsl_permutation_inverse(n_permutation_inv, n_permutation);
      gsl_permute_vector(n_permutation, current_data);
      if (!ppca) {
        gsl_permute_vector(n_permutation, sigma);
        gsl_permute_vector(n_permutation, sigma_inv);
      }
      // go through all the factor analyzers and get their h weights
      prl::log_t<double> log_sum_h; // initialized to represent log(0)
      for (std::size_t j = 0; j < k; ++j) {
        permute_parameters(j, n_permutation);
        // First compute the LU decomposition of M, M_inv, and q for the
        // identified mixture component.
        this->compute_M(j, num_hidden, true); 
        this->compute_q(j, current_data, num_hidden, true); 
        //Compute the addition of the log likelihood and the h accumulator
	prl::log_t<double> prior(gsl_vector_get(pi, j));
	prl::log_t<double> 
          likelihood(log_likelihood(j, current_data, num_hidden, true),
                     prl::log_tag_t());
	prl::log_t<double> posterior =  prior * likelihood;
        fa_factor_scratch_vec[j]->posterior = posterior;
        log_sum_h += posterior;
      }
      current_log_likelihood += log_sum_h.get_log_value();
      // go through all the factor analyzers and compute/accumlate
      // sufficient statistics based on the weights computed above
      for (std::size_t j = 0; j < k; ++j) {
        //stores the sufficient statistics
        compute_sufficient_statistics(j, current_data, num_hidden);
        //unpermutes paramters/statistics so we can accumulate them
        permute_parameters(j, n_permutation_inv);
        permute_sufficient_statistics(n_permutation_inv);
        // accumulate the statistics. Note that this messes up the
        // sufficient statistics, but since we won't need them any
        // more, its ok. Normalize the h values in log space to prevent
        // underflow.
        prl::log_t<double> log_h_norm =
	  fa_factor_scratch_vec[j]->posterior / log_sum_h;
        // The cast below leaves log-space by exponentiation.
        accumulate_sufficient_statistics(j, static_cast<double>(log_h_norm));
      }
      //unpermute sigma
      if (!ppca) {
        gsl_permute_vector(n_permutation_inv, sigma);
        gsl_permute_vector(n_permutation_inv, sigma_inv);
      }
    }
    // done going through all data, now check to see if we have 
    // converged, need to un-relax, or update parameters
    printf("Log likelihood = %14.6lf (difference = %12.6lf) ", 
           current_log_likelihood, 
           current_log_likelihood - previous_log_likelihood); 
    if ((current_log_likelihood + tol < previous_log_likelihood) && 
        (eta > 1.0)) {
      printf("eta reset\n");
      reset_over_relaxation();
    } else {      
      if (current_log_likelihood - previous_log_likelihood 
	  < tol) {
        if (eta > 1.0) 
          eta = 1.0;
        else {
          printf("CONVERGED!\n");
          converged = true;
          break;
        }
      } else 
        eta *= alpha;
      printf("eta: %.3lf ", eta);
      previous_log_likelihood = current_log_likelihood;
      // and re-estimate the parameters for each factor analyzer
      for (std::size_t j = 0; j < k; ++j)  
        recompute_parameters(j, N, eta); 
      recompute_sigma(N, eta);
      // if we have a mixture, show the factor weights
      if (k > 1) {
        printf("pi: ");
        for (std::size_t i = 0; i < pi->size; ++i)
          printf("%.3lf ", gsl_vector_get(pi, i));
      }
      // visual feedback
      if (ppca)
        printf("sigma: %.10lf\n", sqrt(gsl_vector_get(sigma, 0)));
      else {
        if (n < 10) {
          printf("sigma: ");
          for (std::size_t i = 0; i < n; ++i)
            printf("%.3lf ", gsl_vector_get(sigma, i));
        }
        printf("\n");
      }
    }
  }
  return converged;
}

void mfa_t::ppca_solve(vector_data_set_t& data) {
  //we must be initialized
  assert(initialized);
  //only works for ppca
  assert(ppca);
  //only works for one mixture component
  assert(k == 1);
  fa_info_t* fa_info_ptr = this->fa_ptr_vec[0];
  gsl_vector* mu = fa_info_ptr->mu;
  std::size_t num_data_points = 0;
  gsl_vector_set_zero(mu);
  data.reset();
  printf("Computing average..."); fflush(stdout);
  while(data.get_next_vector(current_data)) {     
    num_data_points++;
    gsl_blas_daxpy(1.0, current_data, mu);
  }
  gsl_blas_dscal(1.0/static_cast<double>(num_data_points), mu);
  mu_initialized = true;
  printf("done!\n");
  //allocate data needed once
  gsl_matrix* at_a_matrix    = gsl_matrix_alloc(num_data_points, 
                                                num_data_points);
  gsl_vector* reference_data = gsl_vector_alloc(n);
  //compute A'A where A is the matrix with all the data points
  //this uses a two pass technique instead of storing everything
  //in memory.  For a one pass technique, see teh ppca_bg_model
  printf("Computing A'A\n");
  double dot_prod;
  for (std::size_t frame1_no = 0; frame1_no < num_data_points; ++frame1_no) {
    printf("On row %u of %u\n", frame1_no + 1, num_data_points);
    data.reset();
    for (std::size_t i = 0; i < frame1_no; ++i)
      data.get_next_vector(reference_data);
    data.get_next_vector(reference_data);
    gsl_blas_daxpy(-1.0, mu, reference_data);
    gsl_blas_ddot(reference_data, reference_data, &dot_prod);
    gsl_matrix_set(at_a_matrix, frame1_no, frame1_no, dot_prod);
    for (std::size_t frame2_no = frame1_no + 1; frame2_no < num_data_points; 
         ++frame2_no) {
      data.get_next_vector(current_data);
      gsl_blas_daxpy(-1.0, mu, current_data);
      gsl_blas_ddot(current_data, reference_data, &dot_prod);
      gsl_matrix_set(at_a_matrix, frame1_no, frame2_no, dot_prod);
      gsl_matrix_set(at_a_matrix, frame2_no, frame1_no, dot_prod);
    }
  }
  ppca_svd_solve(at_a_matrix, data, num_data_points);
  //free memory
  gsl_matrix_free(at_a_matrix);
  gsl_vector_free(reference_data);
}

void mfa_t::ppca_solve_fast(vector_data_set_t& data) {
  //we must be initialized
  assert(initialized);
  //only works for ppca
  assert(ppca);
  //only works for one mixture component
  assert(k == 1);
  fa_info_t* fa_info_ptr = this->fa_ptr_vec[0];
  gsl_vector* mu = fa_info_ptr->mu;
  std::size_t num_data_points = 0;
  data.reset();
  gsl_vector_set_zero(mu);
  printf("Computing average..."); fflush(stdout);
  while(data.get_next_vector(current_data)) {     
    gsl_blas_daxpy(1.0, current_data, mu);
    num_data_points++;
  }
  gsl_blas_dscal(1.0 / static_cast<double>(num_data_points), mu);
  printf("done!\n");
  mu_initialized = true;
  printf("Constructing A\n");
  gsl_matrix* A = gsl_matrix_alloc(n, num_data_points);
  data.reset();
  for (std::size_t frame_no = 0; frame_no < num_data_points; ++frame_no) {
    data.get_next_vector(current_data);
    gsl_blas_daxpy(-1.0, mu, current_data);
    gsl_vector_view v = gsl_matrix_column(A, frame_no);
    gsl_blas_dcopy(current_data, &v.vector);
  }
  printf("Computing A'A\n");
  gsl_matrix* at_a_matrix = gsl_matrix_alloc(num_data_points,
                                             num_data_points);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, A, A, 0.0, at_a_matrix);
  ppca_svd_solve(at_a_matrix, data, num_data_points);
  //free memory
  gsl_matrix_free(A);
  gsl_matrix_free(at_a_matrix);
}

bool mfa_t::save(const std::string& path) 
  const throw(std::runtime_error) {
  assert(initialized);
  FILE* outfile;
  if ((outfile = fopen(path.c_str(), "wb"))==NULL)
    throw std::runtime_error("Cannot open file for writing");
  // write the dimensions to file.
  fwrite(&n, sizeof(std::size_t), 1, outfile);
  fwrite(&m, sizeof(std::size_t), 1, outfile);
  fwrite(&k, sizeof(std::size_t), 1, outfile);
  fwrite(&ppca, sizeof(bool), 1, outfile);
  // write the data to file
  assert(pi->size == this->k);
  if (gsl_vector_fwrite(outfile, pi) == GSL_EFAILED)
    throw std::runtime_error("Cannot write pi to file");
  assert(sigma->size == this->n && !ppca ||
         sigma->size == 1 && ppca);
  if (gsl_vector_fwrite(outfile, sigma) == GSL_EFAILED)
    throw std::runtime_error("Cannot write noise variance to file");
  for (std::size_t i = 0; i < k; ++i) {
    fa_info_t* fa_info_ptr = this->fa_ptr_vec[i];
    gsl_vector* mu = fa_info_ptr->mu;
    gsl_matrix* W = fa_info_ptr->W;
    assert(mu->size == n);
    if (gsl_vector_fwrite(outfile, mu) == GSL_EFAILED)
      throw std::runtime_error("Cannot write average data to file");
    assert(W->size1 == n && W->size2 == m);
    if (gsl_matrix_fwrite(outfile, W) == GSL_EFAILED)
      throw std::runtime_error("Cannot write weight matrix to file");
  }
  fclose(outfile);
  return true;
}

void mfa_t::load(const std::string& path) throw(std::runtime_error) {
  FILE* infile;
  if ((infile = fopen(path.c_str(), "rb")) == NULL) 
    throw std::runtime_error("Cannot open file for reading");
  // Read the dimensions from the file.
  fread(&n, sizeof(std::size_t), 1, infile);
  fread(&m, sizeof(std::size_t), 1, infile);
  fread(&k, sizeof(std::size_t), 1, infile);
  fread(&ppca, sizeof(bool), 1, infile);
  initialize(k, m, n, ppca);
  // Read the data from file.
  assert(pi->size == this->k);
  if (gsl_vector_fread(infile, pi) == GSL_EFAILED) 
    throw std::runtime_error("Cannot read pi from file");
  assert(sigma->size == this->n && !ppca ||
         sigma->size == 1 && ppca);
  if (gsl_vector_fread(infile, sigma) == GSL_EFAILED) 
    throw std::runtime_error("Cannot read noise variance from file");
  for (std::size_t i = 0; i < k; ++i) {
    fa_info_t* fa_info_ptr = this->fa_ptr_vec[i];
    gsl_vector* mu = fa_info_ptr->mu;
    gsl_matrix* W = fa_info_ptr->W;
    assert(mu->size == n);
    if (gsl_vector_fread(infile, mu) == GSL_EFAILED) 
      throw std::runtime_error("Cannot read average data from file");
    assert(W->size1 == n && W->size2 == m);
    if (gsl_matrix_fread(infile, W) == GSL_EFAILED) 
      throw std::runtime_error("Cannot read weight matrix from file");
  }
  fclose(infile);
  mu_initialized = true;
  gsl_vector_set_all(sigma_inv, 1.0);
  gsl_vector_div(sigma_inv, sigma);
}

void mfa_t::convert_PPCA_to_FA() {
  if (!ppca)
    return;
  double ppca_sigma = gsl_vector_get(sigma, 0);
  //free the old sigma
  gsl_vector_free(this->sigma);
  gsl_vector_free(this->sigma_inv);
  //and allocate the new sigma
  this->sigma = gsl_vector_alloc(n);
  this->sigma_inv = gsl_vector_alloc(n);
  //and set every element equal to the old ppca sigma
  gsl_vector_set_all(sigma, ppca_sigma);
  gsl_vector_set_all(sigma_inv, 1.0);
  gsl_vector_div(sigma_inv, sigma);
  ppca = false;
}

void mfa_t::convert_FA_to_PPCA() {
  if (ppca)
    return;
  double ppca_sigma = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    ppca_sigma += gsl_vector_get(sigma, i);
  ppca_sigma /= n;
  //free the old sigma
  gsl_vector_free(this->sigma);
  gsl_vector_free(this->sigma_inv);
  //and allocate the new sigma
  this->sigma = gsl_vector_alloc(1);
  this->sigma_inv = gsl_vector_alloc(1);
  //and set the new sigma equal to the average of all the old sigma
  gsl_vector_set(sigma, 0, ppca_sigma);
  gsl_vector_set(sigma_inv, 0, 1.0/ppca_sigma);
  ppca = true;
}

void mfa_t::get_expected_data(const gsl_vector* data,
                                     const std::vector<bool>& hidden_mask,
                                     gsl_vector* expected_data) {
  if (data->size != n) {
    printf("Error: input (size %u) is not the expected size %u\n",
           data->size, n);
    exit(EXIT_FAILURE);
  }
  //get the permutation and permute data
  gsl_blas_dcopy(data, current_data);
  std::size_t num_hidden = compute_permutation(n_permutation, hidden_mask);
  gsl_permutation_inverse(n_permutation_inv, n_permutation);
  gsl_permute_vector(n_permutation, current_data);
  
  if (!ppca) {
    gsl_permute_vector(n_permutation, sigma);
    gsl_permute_vector(n_permutation, sigma_inv);
  }
  //Note: this computes all expected images but just returns the one
  //of the last factor
  for (std::size_t j = 0; j < k; ++j) {
    gsl_vector* mu   = fa_ptr_vec[j]->mu;
    gsl_matrix* W    = fa_ptr_vec[j]->W;
    permute_parameters(j, n_permutation);
    this->compute_M(j, num_hidden); 
    this->compute_q(j, current_data, num_hidden); 
    //now that we have our M and q, unpermute the paramters
    //gets the expected_x
    gsl_linalg_LU_solve(this->M_lu, this->m_permutation, this->q, expected_x);
    //gets the expected data = W<x>+\mu
    permute_parameters(j, n_permutation_inv);
    gsl_blas_dgemv(CblasNoTrans, 1.0, W, expected_x, 0.0, expected_data);
    gsl_vector_add(expected_data, mu);
  }
  // unpermute sigma
  if (!ppca) {
    gsl_permute_vector(n_permutation_inv, sigma);
    gsl_permute_vector(n_permutation_inv, sigma_inv);
  }
}

void mfa_t::print_W_to_file(const std::size_t j,
                                   const std::string& file_name) const {
  assert(initialized);
  assert(j < k);
  FILE *outfile = fopen(file_name.c_str(), "w");
  if (outfile == NULL) {
    printf("ERROR: could not open file %s for writing\n", file_name.c_str());
    exit(EXIT_FAILURE);
  }
  gsl_matrix* W  = fa_ptr_vec[j]->W;
  for (std::size_t r = 0; r < n; ++r) {
    for (std::size_t c = 0; c < m; ++c)
      fprintf(outfile, "%.8lf ", gsl_matrix_get(W, r, c));
    fprintf(outfile, "\n");
  }
  fclose(outfile);
}

void mfa_t::print_mu_to_file(const std::size_t j,
                                    const std::string& file_name) const {
  assert(initialized);
  assert(j < k);
  FILE *outfile = fopen(file_name.c_str(), "w");
  if (outfile == NULL) {
    printf("ERROR: could not open file %s for writing\n", file_name.c_str());
    exit(EXIT_FAILURE);
  }
  gsl_vector* mu  = fa_ptr_vec[j]->mu;
  for (std::size_t i = 0; i < n; ++i) 
    fprintf(outfile, "%.8lf\n", gsl_vector_get(mu, i));
  fclose(outfile);
}

void mfa_t::print_sigma_to_file(const std::string& file_name) const {
  assert(initialized);
  FILE *outfile = fopen(file_name.c_str(), "w");
  if (outfile == NULL) {
    printf("ERROR: could not open file %s for writing\n", file_name.c_str());
    exit(EXIT_FAILURE);
  }
  if (ppca) {
    fprintf(outfile, "%.8lf\n", gsl_vector_get(sigma, 0));
  } else {
    for (std::size_t i = 0; i < n; ++i) 
      fprintf(outfile, "%.8lf\n", gsl_vector_get(sigma, i));
  }
  fclose(outfile);
}

//======================== mfa_t protected ==================================
void mfa_t::initialize(std::size_t k, std::size_t m, 
                              std::size_t n, bool ppca) {
  prng.seed(3);
  this->k = k;
  this->m = m;
  this->n = n;
  this->ppca = ppca;
  // free any memory allocated
  if (initialized) {
    gsl_matrix_free(this->M);
    gsl_matrix_free(this->M_lu);
    gsl_matrix_free(this->M_inv);
    gsl_matrix_free(this->expected_xx);
    gsl_matrix_free(this->m_by_m_scratch);
    gsl_matrix_free(this->n_by_m_scratch);
    gsl_vector_free(this->q);
    gsl_vector_free(this->expected_x);
    gsl_vector_free(this->m_by_1_scratch);
    gsl_vector_free(this->n_by_1_scratch);
    gsl_vector_free(this->current_data);
    gsl_vector_free(this->sigma);
    gsl_vector_free(this->sigma_inv);
    gsl_permutation_free(this->m_permutation);
    gsl_permutation_free(this->n_permutation);
    gsl_permutation_free(this->n_permutation_inv);
    gsl_vector_free(this->pi);
    for (std::size_t i = 0; i < k; ++i)
      delete fa_ptr_vec[i];
  }
  // allocate memory
  this->M              = gsl_matrix_alloc(m, m);
  this->M_lu           = gsl_matrix_alloc(m, m);
  this->M_inv          = gsl_matrix_alloc(m, m);
  this->expected_xx    = gsl_matrix_alloc(m, m);
  this->m_by_m_scratch = gsl_matrix_alloc(m, m);
  this->n_by_m_scratch = gsl_matrix_alloc(n, m);
  this->q              = gsl_vector_alloc(m);
  this->expected_x     = gsl_vector_alloc(m);
  this->m_by_1_scratch = gsl_vector_alloc(m);
  this->n_by_1_scratch = gsl_vector_alloc(n);
  this->current_data   = gsl_vector_alloc(n);
  if (ppca) {
    this->sigma = gsl_vector_alloc(1);
    this->sigma_inv = gsl_vector_alloc(1);
  }
  else {
    this->sigma = gsl_vector_alloc(n);
    this->sigma_inv = gsl_vector_alloc(n);
  }
  this->m_permutation = gsl_permutation_alloc(m);
  this->n_permutation = gsl_permutation_alloc(n);
  this->n_permutation_inv = gsl_permutation_alloc(n);
  gsl_permutation_init(this->m_permutation);
  gsl_permutation_init(this->n_permutation);
  gsl_permutation_init(this->n_permutation_inv);
  // allocate everything for each factor analyzer
  fa_ptr_vec.resize(k, NULL);
  fa_factor_scratch_vec.resize(k, NULL);
  this->pi = gsl_vector_alloc(k);
  gsl_vector_set_all(this->pi, 1.0/static_cast<double>(k));
  // initialize the memory for all the factor analyzers
  for (std::size_t i = 0; i < k; ++i) {
    fa_ptr_vec[i] = new fa_info_t(m, n);
    gsl_vector_set_zero(fa_ptr_vec[i]->mu);
    //add small noise to all the W parameters
    if (k > 1) {
      for (std::size_t row = 0; row < n; row++) 
        for (std::size_t col = 0; col < m; col++)
          gsl_matrix_set(fa_ptr_vec[i]->W, row, col, 1.0 + sample_normal(1.0));
    } else 
      gsl_matrix_set_all(fa_ptr_vec[i]->W, 1.0);
  }
  gsl_vector_set_all(sigma, 1.0);
  gsl_vector_set_all(sigma_inv, 1.0);
  mu_initialized = false;
  previous_log_likelihood = -std::numeric_limits<double>::infinity();
  initialized = true;
}

void mfa_t::ppca_svd_solve(gsl_matrix* at_a_matrix, 
                                  vector_data_set_t& data,
                                  const std::size_t num_data_points) {
  fa_info_t* fa_info_ptr = this->fa_ptr_vec[0];
  gsl_vector* mu = fa_info_ptr->mu;
  gsl_matrix* W  = fa_info_ptr->W;
  //compute the svd of A'A to get eigen vector of A'A
  printf("Computing SVD of A'A\n");
  gsl_matrix* U           = gsl_matrix_alloc(num_data_points, num_data_points);
  gsl_vector* eigen_vals  = gsl_vector_alloc(num_data_points);
  gsl_vector* work_space  = gsl_vector_alloc(num_data_points);
  gsl_linalg_SV_decomp(at_a_matrix, U, eigen_vals, work_space);
  gsl_blas_dscal(1.0 / static_cast<double>(num_data_points), eigen_vals);
  //computing eigenvectors of AA' from those of A'A
  printf("Computing eigenvectors of AA'...\n");
  gsl_matrix_set_zero(W);
  data.reset();
  gsl_matrix_view U_eigen = gsl_matrix_submatrix(U, 0, 0, num_data_points, m);
  for (std::size_t frame_no = 0; frame_no < num_data_points; ++frame_no) {
    data.get_next_vector(current_data);
    gsl_blas_daxpy(-1.0, mu, current_data);
    gsl_vector_view U_row = gsl_matrix_row(&U_eigen.matrix, frame_no);
    gsl_blas_dger(1.0, current_data, &U_row.vector, W);
  }
  //normalize the eigenvectors
  printf("Normalizing eigenvectors of AA'\n");
  for (std::size_t evec_index = 0; evec_index < m; ++evec_index) {
    gsl_vector_view evec = gsl_matrix_column(W, evec_index);
    double norm = gsl_blas_dnrm2(&evec.vector);
    gsl_blas_dscal(1.0/norm, &evec.vector);
  }
  //compute sigma^2
  double sum = 0.0;
  for (std::size_t i = m; i < num_data_points; ++i)
    sum += gsl_vector_get(eigen_vals, i);
  sum /= static_cast<double>(n - m);
  gsl_vector_set(sigma, 0, sum);
  gsl_vector_set(sigma_inv, 0, 1.0/sum);
  //compute final W matrix
  for (std::size_t i = 0; i < m; ++i) {
    double scalar = sqrt(gsl_vector_get(eigen_vals, i) - sum);
    gsl_vector_view W_col = gsl_matrix_column(W, i);
    gsl_blas_dscal(scalar, &W_col.vector);
  }
  gsl_matrix_free(U);
  gsl_vector_free(eigen_vals);
  gsl_vector_free(work_space);
  printf("Done!\n");
}

std::size_t mfa_t::compute_permutation(gsl_permutation* permutation, 
                                              const gsl_vector* data) {
  //make sure we have the right sized data
  assert(data->size == permutation->size);
  //and initialize the permutation (0, 1, ..., d)
  gsl_permutation_init(permutation);
  int obs_tail = permutation->size-1;
  //have obs_tail point to the first element from the end that is hidden
  while (obs_tail >= 0 && !isnan(gsl_vector_get(data, obs_tail))) {
    obs_tail--;
  }
  if (obs_tail < 0)
    return 0;
  if (obs_tail == 0)
    return 1;
  int num_hidden = 0;
  while(true) {
    const double value = gsl_vector_get(data, num_hidden);
    if (!isnan(value)) {
      gsl_permutation_swap(permutation, num_hidden, obs_tail);
      num_hidden++;
      obs_tail--;
      while (obs_tail > num_hidden
             && !isnan(gsl_vector_get(data, obs_tail))) {
        obs_tail--;
      }
    } else
      num_hidden++;
    if (obs_tail <= num_hidden) {
      if (num_hidden < static_cast<int>(data->size) && isnan(num_hidden)) 
        return static_cast<std::size_t>(num_hidden + 1); 
      else
        return static_cast<std::size_t>(num_hidden); 
    }
  }
}

std::size_t mfa_t::compute_permutation(gsl_permutation* permutation, 
                                              const std::vector<bool>& hidden_mask) {
  //make sure we have the right sized mask
  assert(hidden_mask.size() == permutation->size);
  //and initialize the permutation (0, 1, ..., d)
  gsl_permutation_init(permutation);
  int obs_tail = permutation->size-1;
  //have obs_tail point to the first element from the end that is hidden
  while (obs_tail >= 0 && !hidden_mask[obs_tail])
    obs_tail--;
  if (obs_tail < 0)
    return 0;
  if (obs_tail == 0)
    return 1;
  int num_hidden = 0;
  while(true) {
    if (!hidden_mask[num_hidden]) {
      gsl_permutation_swap(permutation, num_hidden, obs_tail);
      num_hidden++;
      obs_tail--;
      while (obs_tail > num_hidden && !hidden_mask[obs_tail])
        obs_tail--;
    } else
      num_hidden++;
    if (obs_tail <= num_hidden) {
      if (num_hidden < static_cast<int>(hidden_mask.size()) 
          && hidden_mask[num_hidden])
        return static_cast<std::size_t>(num_hidden + 1); 
      else
        return static_cast<std::size_t>(num_hidden); 
    }
  }
}

void mfa_t::permute_parameters(const std::size_t j,
                                      const gsl_permutation* permutation) {
  fa_info_t* fa_info_ptr = this->fa_ptr_vec[j];
  gsl_vector* mu = fa_info_ptr->mu;
  gsl_matrix* W = fa_info_ptr->W;
  gsl_vector_view n_vector;
  gsl_permute_vector(permutation, mu);
  for (std::size_t i = 0; i < W->size2; ++i) {
    n_vector = gsl_matrix_column(W, i);
    gsl_permute_vector(permutation, &n_vector.vector);
  }
}

void 
mfa_t::permute_sufficient_statistics(const gsl_permutation* permutation) {
  gsl_vector* expected_y  = mfa_scratch_ptr->expected_y;
  gsl_matrix* expected_yx = mfa_scratch_ptr->expected_yx;
  gsl_vector* expected_yy = mfa_scratch_ptr->expected_yy;
  gsl_vector_view n_vector;
  gsl_permute_vector(permutation, expected_y);
  gsl_permute_vector(permutation, expected_yy);
  for (std::size_t i = 0; i < m; ++i) {
    n_vector = gsl_matrix_column(expected_yx, i);
    gsl_permute_vector(permutation, &n_vector.vector);
  }
}


void mfa_t::compute_mu(vector_data_set_t& data) {
  //since this is only called once, we allocate some memory
  gsl_vector* next_vector = gsl_vector_alloc(n);
  gsl_vector* count = gsl_vector_alloc(n);
  gsl_vector_set_zero(current_data);
  gsl_vector_set_zero(count);
  data.reset();
  //accumlates all the data points and keeps track of how many 
  //times each data point is observed
  while(data.get_next_vector(next_vector)) {
    for (std::size_t i = 0; i < n; ++i) {
      double next_data = gsl_vector_get(next_vector, i);
      if (isnan(next_data))
        continue;
      gsl_vector_set(current_data, i, 
                     gsl_vector_get(current_data, i) + next_data);
      gsl_vector_set(count, i,
                     gsl_vector_get(count, i) + 1.0);
    }
  }
  gsl_vector_div(current_data, count);
  //checks to see if any data points were never observed.
  //if so, it sets them to be 0
  for (std::size_t i = 0; i < n; ++i) {
    double cur_data_pt = gsl_vector_get(current_data, i);
    if (isnan(cur_data_pt)) {
      gsl_vector_set(current_data, i, 0);
      assert(gsl_vector_get(count, i) == 0);
    }
  }
  //free memory allocated
  gsl_vector_free(next_vector);
  gsl_vector_free(count);
  mu_initialized = true;
}

void mfa_t::compute_M(const std::size_t j,
                             const std::size_t num_hidden,
                             const bool to_scratch) const {
  // M = I + W_o' \Sigma^{-1} W_o
  fa_info_t* fa_info_ptr = fa_ptr_vec[j];
  gsl_matrix* W          = fa_info_ptr->W;
  gsl_matrix *M_j, *M_j_lu, *M_j_inv;
  gsl_permutation *mj_permutation;
  //determines where to store the calculations
  if (to_scratch) {
    fa_factor_scratch_t* fa_factor_scratch_ptr = fa_factor_scratch_vec[j];
    M_j     = fa_factor_scratch_ptr->M_j;
    M_j_lu  = fa_factor_scratch_ptr->M_j_lu;
    M_j_inv = fa_factor_scratch_ptr->M_j_inv;
    mj_permutation = fa_factor_scratch_ptr->m_permutation;
  } else {
    M_j = this->M;
    M_j_lu = this->M_lu;
    M_j_inv = this->M_inv;
    mj_permutation = this->m_permutation;
  }
  gsl_matrix_view W_obs 
    = gsl_matrix_submatrix(W, num_hidden, 0, n - num_hidden, m);
  if (ppca) { 
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0,
                   &W_obs.matrix, &W_obs.matrix, 0.0, M_j);
    double s = gsl_vector_get(sigma_inv, 0);
    gsl_matrix_scale(M_j, s);
    for (std::size_t i = 0; i < m; ++i)
      gsl_matrix_set(M_j, i, i, gsl_matrix_get(M_j, i, i) + 1.0);
  } else {
    gsl_matrix_view obs_by_m_scratch
      = gsl_matrix_submatrix(n_by_m_scratch, num_hidden, 0, n - num_hidden, m);
    gsl_matrix_memcpy(&obs_by_m_scratch.matrix, &W_obs.matrix);
    //This speeds things up much more than doing a vector view and then
    //division for some reason
    for (std::size_t row = num_hidden; row < n; ++row) {
      double s = gsl_vector_get(sigma_inv, row);
      for (std::size_t col = 0; col < m; ++col) {
        gsl_matrix_set(n_by_m_scratch, row, col,
                       gsl_matrix_get(n_by_m_scratch, row, col) * s);
      }
    }
    gsl_matrix_set_identity(M_j);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, &obs_by_m_scratch.matrix,
                   &W_obs.matrix, 1.0, M_j);
  }
  int sign;
  gsl_matrix_memcpy(M_j_lu, M_j);
  gsl_linalg_LU_decomp(M_j_lu, mj_permutation, &sign);
  gsl_linalg_LU_invert(M_j_lu, mj_permutation, M_j_inv);
  if (to_scratch)
    fa_factor_scratch_vec[j]->sign = sign;
  else
    this->sign = sign;
}

void mfa_t::compute_q(const std::size_t j,
			     const gsl_vector* data,
                             const std::size_t num_hidden,
                             const bool to_scratch) const {
  fa_info_t* fa_info_ptr = this->fa_ptr_vec[j];
  gsl_vector* mu = fa_info_ptr->mu;
  gsl_matrix* W  = fa_info_ptr->W;
  gsl_vector* q_j;
  if (to_scratch)
    q_j = fa_factor_scratch_vec[j]->q_j;
  else 
    q_j = this->q;
  gsl_matrix_view W_obs 
    = gsl_matrix_submatrix(W, num_hidden, 0, n - num_hidden, m);
  gsl_vector_view obs_by_1_scratch = 
    gsl_vector_subvector(n_by_1_scratch, num_hidden, n - num_hidden);
  gsl_vector_view mu_obs = 
    gsl_vector_subvector(mu, num_hidden, n - num_hidden);
  gsl_vector_const_view data_obs = 
    gsl_vector_const_subvector(data, num_hidden, n - num_hidden);
  gsl_blas_dcopy(&data_obs.vector, &obs_by_1_scratch.vector);
  gsl_vector_sub(&obs_by_1_scratch.vector, &mu_obs.vector);
  if (ppca) {
    gsl_blas_dgemv(CblasTrans, 1.0, &W_obs.matrix, &obs_by_1_scratch.vector,
                   0.0, q_j);
    double s = gsl_vector_get(sigma_inv, 0); 
    gsl_blas_dscal(s, q_j);
  } else {
    for (std::size_t row = num_hidden; row < n; ++row) {
      double s = gsl_vector_get(sigma_inv, row);
      gsl_vector_set(n_by_1_scratch, row, 
                     gsl_vector_get(n_by_1_scratch, row) * s);
    }
    gsl_blas_dgemv(CblasTrans, 1.0, &W_obs.matrix, &obs_by_1_scratch.vector,
                   0.0, q_j);
  }
}

void 
mfa_t::compute_sufficient_statistics(const std::size_t j, 
                                            const gsl_vector* data,
                                            const std::size_t num_hidden) {
  //requires that M_lu, M_inv and q have been computed
  gsl_matrix* M_j_lu  = fa_factor_scratch_vec[j]->M_j_lu;
  gsl_matrix* M_j_inv = fa_factor_scratch_vec[j]->M_j_inv;
  gsl_vector* q_j     = fa_factor_scratch_vec[j]->q_j;
  gsl_permutation* mj_permutation = fa_factor_scratch_vec[j]->m_permutation;
  fa_info_t* fa_info_ptr  = fa_ptr_vec[j];
  gsl_vector* mu          = fa_info_ptr->mu;
  gsl_matrix* W           = fa_info_ptr->W;
  gsl_vector* expected_y  = mfa_scratch_ptr->expected_y;
  gsl_vector* expected_yy = mfa_scratch_ptr->expected_yy;
  gsl_matrix* expected_yx = mfa_scratch_ptr->expected_yx;
  // Now compute <x> = (M_inv * q^T).  This uses the LU solver,
  // rather than computing the inverse explicitly (which is more
  // expensive and numerically unstable).
  gsl_linalg_LU_solve(M_j_lu, mj_permutation, q_j, expected_x);
  // And compute <xx^T> = M_inv + <x><x>^T
  gsl_matrix_memcpy(expected_xx, M_j_inv);
  gsl_blas_dger(1.0, expected_x, expected_x, expected_xx);
  //copy the data into expected data since we won't re-estimate the
  //observed data
  gsl_vector_const_view data_obs =
    gsl_vector_const_subvector(data, num_hidden, n - num_hidden);
  gsl_vector_view expected_y_obs = 
    gsl_vector_subvector(expected_y, num_hidden, n - num_hidden);
  gsl_vector_view expected_yy_obs = 
    gsl_vector_subvector(expected_yy, num_hidden, n - num_hidden);
  gsl_blas_dcopy(&data_obs.vector, &expected_y_obs.vector);
  gsl_blas_dcopy(&expected_y_obs.vector, &expected_yy_obs.vector);
  gsl_vector_mul(&expected_yy_obs.vector, &expected_y_obs.vector);
  
  gsl_matrix_view expected_yx_obs = 
    gsl_matrix_submatrix(expected_yx, num_hidden, 0, n - num_hidden, m);
  gsl_matrix_set_zero(&expected_yx_obs.matrix);
  if (num_hidden > 0) {
    gsl_matrix_view W_hidden 
      = gsl_matrix_submatrix(W, 0, 0, num_hidden, m);
    gsl_vector_view mu_hidden = 
      gsl_vector_subvector(mu, 0, num_hidden);
    gsl_vector_view expected_y_hidden = 
      gsl_vector_subvector(expected_y, 0, num_hidden);
    gsl_blas_dcopy(&mu_hidden.vector, &expected_y_hidden.vector);
    gsl_blas_dgemv(CblasNoTrans, 1.0, &W_hidden.matrix, expected_x, 
                   1.0, &expected_y_hidden.vector);
    gsl_matrix_view expected_yx_hidden = 
      gsl_matrix_submatrix(expected_yx, 0, 0, num_hidden, m);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, 
                   &W_hidden.matrix, M_j_inv,
                   0.0, &expected_yx_hidden.matrix);
    //<yy^T> = sigma + W_i * <xx^t> * W_i' + 2 * W_i * <x> * mu_i + mu_i^2
    double s = gsl_vector_get(sigma, 0); // for PPCA case
    double a, b, mu_i;
    gsl_vector_view w_row;
    for (std::size_t i = 0; i < num_hidden; ++i) {
      w_row = gsl_matrix_row(W, i);
      mu_i = gsl_vector_get(mu, i);
      gsl_blas_ddot(&w_row.vector, expected_x, &a);
      gsl_blas_dgemv(CblasNoTrans, 1.0, expected_xx, 
                     &w_row.vector, 0.0, m_by_1_scratch);
      gsl_blas_ddot(&w_row.vector, m_by_1_scratch, &b);
      if (!ppca)
        s = gsl_vector_get(sigma, i);
      gsl_vector_set(expected_yy, i, s + b + 2.0 * a * mu_i + mu_i * mu_i);
    } 
  }
  // Let <yx^T> += <y><x>^T
  gsl_blas_dger(1.0, expected_y, expected_x, expected_yx);
  return;  
}

void 
mfa_t::accumulate_sufficient_statistics(const std::size_t j,
                                               const double h) {
  gsl_matrix* expected_yx = mfa_scratch_ptr->expected_yx;
  gsl_vector* expected_y  = mfa_scratch_ptr->expected_y;
  gsl_vector* expected_yy = mfa_scratch_ptr->expected_yy;
  fa_factor_scratch_t* fa_factor_scratch_ptr = fa_factor_scratch_vec[j];
  gsl_matrix* W1     = fa_factor_scratch_ptr->W1;
  gsl_matrix* W2     = fa_factor_scratch_ptr->W2;
  gsl_vector* sigma1 = fa_factor_scratch_ptr->sigma1;
  //update W1
  gsl_matrix_view W1_yx = gsl_matrix_submatrix(W1, 0, 0, n, m);
  gsl_vector_view W1_y  = gsl_matrix_column(W1, m);
  gsl_matrix_scale(expected_yx, h);
  gsl_matrix_add(&W1_yx.matrix, expected_yx);
  gsl_blas_dscal(h, expected_y);
  gsl_vector_add(&W1_y.vector, expected_y);
  //update W2
  gsl_matrix_view W2_xx = gsl_matrix_submatrix(W2, 0, 0, m, m);
  gsl_matrix_view W2_x  = gsl_matrix_submatrix(W2, 0, m, m, 1);
  gsl_vector_view W2_xv = gsl_matrix_column(&W2_x.matrix, 0);
  gsl_matrix_scale(expected_xx, h);
  gsl_matrix_add(&W2_xx.matrix, expected_xx);
  gsl_blas_dscal(h, expected_x);
  gsl_vector_add(&W2_xv.vector, expected_x);
  //update sigma1
  gsl_blas_dscal(h, expected_yy);
  gsl_vector_add(sigma1, expected_yy);
  //update h
  fa_factor_scratch_ptr->h1 += h;
}

void mfa_t::recompute_parameters(const std::size_t j, 
                                        const std::size_t N, 
                                        const double eta) {
  fa_info_t* fa_info_ptr = fa_ptr_vec[j];
  gsl_matrix* W  = fa_info_ptr->W;
  gsl_vector* mu = fa_info_ptr->mu;
  fa_factor_scratch_t* fa_factor_scratch_ptr = fa_factor_scratch_vec[j];
  gsl_matrix* W1 = fa_factor_scratch_ptr->W1;
  gsl_matrix* W2 = fa_factor_scratch_ptr->W2;
  double h1      = fa_factor_scratch_ptr->h1;
  // make copies of the old parameters if we are performing over-releaxation
  if (eta > 1.0) {
    // again, note that gsl and blas have different src/dest orders
    gsl_matrix_memcpy(n_by_m_scratch, W);
    gsl_blas_dcopy(mu, n_by_1_scratch);
  }
  // recompute \f$W\f$ and \f$\mu\f$
  // fill in the rest of W2 since we don't repeat these computations
  gsl_matrix_view W2_x   = gsl_matrix_submatrix(W2, 0, m, m, 1);
  gsl_vector_view W2_xv  = gsl_matrix_column(&W2_x.matrix, 0);
  gsl_matrix_view W2_xt  = gsl_matrix_submatrix(W2, m, 0, 1, m);
  gsl_vector_view W2_xtv = gsl_matrix_row(&W2_xt.matrix, 0);
  gsl_blas_dcopy(&W2_xv.vector, &W2_xtv.vector);
  gsl_matrix_set(W2, m, m, h1);
  //compute \f$[W; mu] = W1 * W2^{-1}\f$
  int sign;
  gsl_linalg_LU_decomp(W2, mfa_scratch_ptr->m1_permutation, &sign);
  gsl_linalg_LU_invert(W2, mfa_scratch_ptr->m1_permutation, 
                       mfa_scratch_ptr->m1_by_m1_scratch);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,
                 W1, mfa_scratch_ptr->m1_by_m1_scratch, 0.0, 
                 mfa_scratch_ptr->n_by_m1_scratch);
  //copy corresponding parts into \f$W\f$ and \f$\mu\f$
  gsl_matrix_view W_new = 
    gsl_matrix_submatrix(mfa_scratch_ptr->n_by_m1_scratch, 0, 0, n, m);
  gsl_matrix_memcpy(W, &W_new.matrix);
  gsl_vector_view mu_new = 
    gsl_matrix_column(mfa_scratch_ptr->n_by_m1_scratch, m);
  gsl_blas_dcopy(&mu_new.vector, mu);
  // recompute pi_i = h1 / N
  gsl_vector_set(pi, j, h1 / static_cast<double>(N));
  // make copies of the parameters as estimated by EM
  gsl_matrix_memcpy(fa_factor_scratch_ptr->W_em, W);
  gsl_blas_dcopy(mu, fa_factor_scratch_ptr->mu_em);
  // perform over-relaxation if we want to
  if (eta > 1.0) {
    // first for W = \eta * W_new + (1 - \eta) * W_old
    gsl_matrix_scale(W, eta);
    gsl_matrix_scale(n_by_m_scratch, 1.0 - eta);
    gsl_matrix_add(W, n_by_m_scratch);
    // for mu
    gsl_blas_dscal(eta, mu);
    gsl_blas_dscal(1.0 - eta, n_by_1_scratch);
    gsl_vector_add(mu, n_by_1_scratch);
  }
}

void mfa_t::recompute_sigma(const std::size_t N, const double eta) {
  //minimum value to set sigma at.  Since we are restricting the form of 
  //sigma, we may have negative diagonal elements, which is a result of
  //our model being incorrect or a set of bad paramters in EM.
  //TODO: could make this a paramter
  double min_sigma = 0.0001;

  gsl_vector_set_zero(current_data);
  for (std::size_t j = 0; j < k; ++j) {
    fa_factor_scratch_t* fa_factor_scratch_ptr = fa_factor_scratch_vec[j];
    gsl_matrix* W1        = fa_factor_scratch_ptr->W1;
    gsl_vector* sigma1    = fa_factor_scratch_ptr->sigma1;
    gsl_matrix_view W1_yx = gsl_matrix_submatrix(W1, 0, 0, n, m);
    gsl_vector_view W1_y  = gsl_matrix_column(W1, m);
    fa_info_t* fa_info    = fa_ptr_vec[j];
    gsl_matrix* W         = fa_info->W;
    gsl_vector* mu        = fa_info->mu;
    double dot_prod, result;
    gsl_vector_view w1_row, w_new_row;
    for (std::size_t i = 0; i < n; ++i) {
      w1_row    = gsl_matrix_row(&W1_yx.matrix, i);
      w_new_row = gsl_matrix_row(W, i);
      gsl_blas_ddot(&w_new_row.vector, &w1_row.vector, &dot_prod);
      result = gsl_vector_get(sigma1, i) - 
        gsl_vector_get(&W1_y.vector, i) * gsl_vector_get(mu, i) - dot_prod;
      gsl_vector_set(n_by_1_scratch, i,  result);
    }
    gsl_vector_add(current_data, n_by_1_scratch);
  }
  if (ppca) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      sum += gsl_vector_get(current_data, i);
    sum /= static_cast<double>(n * N);
    if (sum < min_sigma)
      gsl_vector_set(sigma, 0, min_sigma);
    else
      gsl_vector_set(sigma, 0, sum);
  } else {
    gsl_blas_dscal(1.0/static_cast<double>(N), current_data);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      double result = gsl_vector_get(current_data, i);
      if (result < min_sigma) {
        gsl_vector_set(sigma, i, min_sigma);
        sum += min_sigma;
      }
      else {
        gsl_vector_set(sigma, i, result);
        sum += result;
      }
    }
    printf(" avg sigma: %lf ", sqrt(sum / static_cast<double>(n)));
  }
  gsl_vector_set_all(sigma_inv, 1.0);
  gsl_vector_div(sigma_inv, sigma);
  //if we are doing over-relaxation, recompute what the sigma would have
  //been with the em un-relaxed parameters
  if (eta > 1.0) {
    gsl_vector_set_zero(current_data);
    for (std::size_t j = 0; j < k; ++j) {
      fa_factor_scratch_t* fa_factor_scratch_ptr = fa_factor_scratch_vec[j];
      gsl_matrix* W1        = fa_factor_scratch_ptr->W1;
      gsl_matrix* W         = fa_factor_scratch_ptr->W_em;
      gsl_vector* mu        = fa_factor_scratch_ptr->mu_em;
      gsl_vector* sigma1    = fa_factor_scratch_ptr->sigma1;
      gsl_matrix_view W1_yx = gsl_matrix_submatrix(W1, 0, 0, n, m);
      gsl_vector_view W1_y  = gsl_matrix_column(W1, m);
      double dot_prod, result;
      gsl_vector_view w1_row, w_new_row;
      for (std::size_t i = 0; i < n; ++i) {
        w1_row    = gsl_matrix_row(&W1_yx.matrix, i);
        w_new_row = gsl_matrix_row(W, i);
        gsl_blas_ddot(&w_new_row.vector, &w1_row.vector, &dot_prod);
        result = gsl_vector_get(sigma1, i) - 
          gsl_vector_get(&W1_y.vector, i) * gsl_vector_get(mu, i) - dot_prod;
        gsl_vector_set(n_by_1_scratch, i,  result);
      }
      gsl_vector_add(current_data, n_by_1_scratch);
    }
    gsl_vector* sigma_em = mfa_scratch_ptr->sigma_em;
    if (ppca) {
      double sum = 0.0;
      for (std::size_t i = 0; i < n; ++i)
        sum += gsl_vector_get(current_data, i);
      sum /= static_cast<double>(n * N);
      if (sum < min_sigma)
        gsl_vector_set(sigma_em, 0, min_sigma);
      else
        gsl_vector_set(sigma_em, 0, sum);
    } else {
      gsl_blas_dscal(1.0/static_cast<double>(N), current_data);
      for (std::size_t i = 0; i < n; ++i) {
        double result = gsl_vector_get(current_data, i);
        if (result < min_sigma)
          gsl_vector_set(sigma_em, i, min_sigma);
        else
          gsl_vector_set(sigma_em, i, result);
      }
    }
  }
}

void mfa_t::reset_over_relaxation() {
  this->eta = 1.0;
  //set all parameters to be their un-relaxed form as estimated by em last step
  gsl_blas_dcopy(mfa_scratch_ptr->sigma_em, sigma);
  gsl_vector_set_all(sigma_inv, 1.0);
  gsl_vector_div(sigma_inv, sigma);
  for (std::size_t j = 0; j < k; ++j) {
    fa_factor_scratch_t* fa_factor_scratch_ptr = fa_factor_scratch_vec[j];
    fa_info_t* fa_info_ptr       = fa_ptr_vec[j];
    //note that gsl and blas are inconsistent with src/dest for copying
    gsl_blas_dcopy(fa_factor_scratch_ptr->mu_em, fa_info_ptr->mu);
    gsl_matrix_memcpy(fa_info_ptr->W, fa_factor_scratch_ptr->W_em);
  }
}

double mfa_t::log_likelihood(const std::size_t j,
				    const gsl_vector* data,
                                    const std::size_t num_hidden,
                                    const bool from_scratch) const {
  // Now compute a = (q * M_inv * q^T).  Start by computing M_inv *
  // q^T.  This uses the LU solver, rather than computing the
  // inverse explicitly (which is more expensive and numerically
  // unstable).
  gsl_matrix* M_j_lu;
  gsl_vector* q_j;
  gsl_permutation* mj_permutation;
  int sign;
  
  if (from_scratch) {
    fa_factor_scratch_t* fa_factor_scratch_ptr = fa_factor_scratch_vec[j];
    M_j_lu         = fa_factor_scratch_ptr->M_j_lu;
    mj_permutation = fa_factor_scratch_ptr->m_permutation;
    q_j            = fa_factor_scratch_ptr->q_j;
    sign           = fa_factor_scratch_ptr->sign;
  } else {
    M_j_lu         = this->M_lu;
    mj_permutation = this->m_permutation;
    q_j            = this->q;
    sign           = this->sign;
  }
  gsl_linalg_LU_solve(M_j_lu, mj_permutation, q_j, this->m_by_1_scratch);
  // Now compute the dot product of that with q.
  double a;
  gsl_blas_ddot(q_j, this->m_by_1_scratch, &a);
  // Now compute b = (y - \mu)^T \Sigma^{-1} (y - \mu), where we look
  // only at components of y that are observed.  At the same time,
  // count the number of observations and compute the log determinant
  // of Sigma.
  double b = 0.0;
  double log_det_Sigma = 0.0;
  gsl_vector* mu = this->fa_ptr_vec[j]->mu;
  gsl_vector_const_view data_obs =
    gsl_vector_const_subvector(data, num_hidden, n - num_hidden);
  gsl_vector_view mu_obs = 
    gsl_vector_subvector(mu, num_hidden, n - num_hidden);
  gsl_vector_view obs_by_1_scratch = 
    gsl_vector_subvector(n_by_1_scratch, num_hidden, n - num_hidden);

  gsl_blas_dcopy(&data_obs.vector, &obs_by_1_scratch.vector);
  gsl_vector_sub(&obs_by_1_scratch.vector, &mu_obs.vector);
  if (ppca) {
    gsl_blas_ddot(&obs_by_1_scratch.vector, &obs_by_1_scratch.vector, &b);
    double s = gsl_vector_get(sigma, 0);
    b /= s;
    log_det_Sigma = (n - num_hidden) * log(s); 
  } else {
    double d, s;
    for (std::size_t i = num_hidden; i < n; ++i) {
      s = gsl_vector_get(sigma, i);
      d = gsl_vector_get(n_by_1_scratch, i);
      b += d * d / s;
      log_det_Sigma += log(s);
    }
  }
  // Now compute the exponent.
  double e = -(b - a) / 2.0;
  // Now compute the log normalizer for the Gaussian.
  double z = -0.5 * (// The constant below is ln(2\pi)
    (n - num_hidden) * 1.837877066409
    // This is log(det(W W^T + \Sigma))
    + log_det_Sigma 
    + log(gsl_linalg_LU_det(M_j_lu, sign)));
  // The overall log likelihood is the sum of z and e.
  return e + z;
}

double mfa_t::sample_normal(const double sigma) {
  // set up the normal distribution
  boost::normal_distribution<double> norm_dist(0.0, sigma);
  
  // bind random number generator to distribution, forming a function
  boost::variate_generator<boost::mt19937&, 
    boost::normal_distribution<double> > 
    normal_sampler(prng, norm_dist);
  
  // sample from the normal distribution
  return normal_sampler();
}

double mfa_t::sample_uniform01() {
  boost::uniform_01<boost::mt19937> uniform_01_rng(prng);
  return uniform_01_rng();
}

