#ifndef kernel_h
#define kernel_h 1

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "globals.h"
#include "example_set.h"
#include "parameters.h"

/**
 * Base class for all kernels
 * @li kernel caching
 *
 * @author Stefan Rueping <rueping@ls8.cs.uni-dortmund.de>
 * @version 0.1
 **/


class kernel_c{
 protected:
  SVMINT cache_access;
  SVMINT cache_misses;
  SVMINT counter;  // time index for last access
  SVMINT cache_size;  // number of rows in cache
  SVMINT cache_mem;   // max. size of memory for cache
  SVMINT examples_size;  // length of a row
  SVMFLOAT** rows;
  SVMINT* last_used; // the heap
  SVMINT* index;
  void clean_cache();

  // little helpers
  SVMFLOAT innerproduct(const svm_example x, const svm_example y);
  SVMFLOAT norm2(const svm_example x, const svm_example y);

  example_set_c *the_examples;
 public:
  SVMINT dim;
  // caching based on i
  friend std::istream& operator >> (std::istream& data_stream, kernel_c& the_kernel);
  friend std::ostream& operator << (std::ostream& data_stream, kernel_c& the_kernel);
  virtual void input(std::istream& data_stream);
  virtual void output(std::ostream& data_stream) const;

  kernel_c();
  virtual ~kernel_c();
  virtual void init(SVMINT new_cache_MB,example_set_c* new_examples);
  void set_examples_size(SVMINT new_examples_size);
  int cached(const SVMINT i);
  int check();
  virtual void overwrite(const SVMINT i, const SVMINT j);
  SVMINT lookup(const SVMINT i);
  virtual SVMFLOAT calculate_K(const SVMINT i, const SVMINT j);
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);
  SVMFLOAT* get_row(const SVMINT i); // returned pointer will not be manipulated
  virtual void compute_row(const SVMINT i, SVMFLOAT* row);
};

std::istream& operator >> (std::istream& data_stream, kernel_c& the_kernel);
std::ostream& operator << (std::ostream& data_stream, kernel_c& the_kernel);


class kernel_dot_c : public kernel_c{
 public:
  kernel_dot_c(){};
  virtual void input(std::istream& data_stream);
  virtual void output(std::ostream& data_stream) const;
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);
};


class kernel_lin_dot_c : public kernel_c{
 protected:
  SVMFLOAT a;
  SVMFLOAT b;
 public:
  kernel_lin_dot_c(){ a=1; b=0; };
  virtual void input(std::istream& data_stream);
  virtual void output(std::ostream& data_stream) const;
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);
};


class kernel_polynomial_c : public kernel_c{
 protected:
  SVMINT degree;
 public:
  kernel_polynomial_c(){};
  virtual void input(std::istream& data_stream);
  virtual void output(std::ostream& data_stream) const;
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);
};


class kernel_radial_c : public kernel_c{
 protected:
  SVMFLOAT gamma;
 public:
  kernel_radial_c(){};
  virtual void input(std::istream& data_stream);
  virtual void output(std::ostream& data_stream) const;
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);
};


class kernel_neural_c : public kernel_c{
 protected:
  SVMFLOAT a,b;
 public:
  kernel_neural_c(){};
  virtual void input(std::istream& data_stream);
  virtual void output(std::ostream& data_stream) const;
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);
};


class kernel_anova_c : public kernel_c{
 protected:
  SVMINT degree;
  SVMFLOAT gamma;
 public:
  kernel_anova_c(){};
  virtual void input(std::istream& data_stream);
  virtual void output(std::ostream& data_stream) const;
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);
};


class kernel_exponential_c : public kernel_c{
 protected:
  SVMFLOAT lambda;
 public:
  kernel_exponential_c(){};
  virtual void input(std::istream& data_stream);
  virtual void output(std::ostream& data_stream) const;
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);
};


class kernel_user_c : public kernel_c{
 protected:
  SVMINT param_i_1, param_i_2, param_i_3, param_i_4, param_i_5;
  SVMFLOAT param_f_1, param_f_2, param_f_3, param_f_4, param_f_5;
 public:
  kernel_user_c();
  virtual void input(std::istream& data_stream);
  virtual void output(std::ostream& data_stream) const;
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);
};


class kernel_user2_c : public kernel_c{
 protected:
  SVMINT number_param;
  SVMINT* param_i;
  SVMFLOAT* param_f;
 public:
  kernel_user2_c();
  ~kernel_user2_c();
  virtual void input(std::istream& data_stream);
  virtual void output(std::ostream& data_stream) const;
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);
};


class kernel_zero_c : public kernel_c{
 public:
  void input(std::istream& data_stream);
  void output(std::ostream& data_stream) const;
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);
};


class kernel_fourier_c : public kernel_c{
 protected:
  SVMINT N;
 public:
  kernel_fourier_c();
  void input(std::istream& data_stream);
  void output(std::ostream& data_stream) const;
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);
};


class kernel_reg_fourier_c : public kernel_c{
 protected:
  SVMFLOAT q;
 public:
  kernel_reg_fourier_c();
  void input(std::istream& data_stream);
  void output(std::ostream& data_stream) const;
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);
};


class kernel_lintransform_c : public kernel_c{
 protected:
  kernel_c* subkernel;
  SVMFLOAT a,b;
 public:
  kernel_lintransform_c();
  ~kernel_lintransform_c();
  void input(std::istream& data_stream);
  void output(std::ostream& data_stream) const;
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);

};


class kernel_subseq_c : public kernel_c{
 protected:
  SVMINT step;
  SVMFLOAT lambda;
  SVMFLOAT gamma;
 public:
  kernel_subseq_c();
  ~kernel_subseq_c();
  virtual void input(std::istream& data_stream);
  virtual void output(std::ostream& data_stream) const;
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);
};


class kernel_discrete_subseq_c : public kernel_c{
 protected:
  //  SVMINT step;
  SVMINT length;
  SVMFLOAT lambda;
  SVMFLOAT diff;
  svm_example the_x;
  svm_example the_y;
  virtual SVMFLOAT calculate_inner_K(const SVMINT size, const SVMINT end_x, const SVMINT end_y);
  virtual SVMFLOAT calculate_inner_K_prime(const SVMINT size, const SVMINT end_x, const SVMINT end_y);
 public:
  kernel_discrete_subseq_c();
  ~kernel_discrete_subseq_c();
  virtual void input(std::istream& data_stream);
  virtual void output(std::ostream& data_stream) const;
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);
};


class kernel_regularized_c : public kernel_c{
 protected:
  SVMFLOAT* cache;
  kernel_c* inner_kernel;
 public:
  kernel_regularized_c();
  ~kernel_regularized_c();
  void input(std::istream& data_stream);
  void output(std::ostream& data_stream) const;
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);
  virtual void compute_row(const SVMINT i, SVMFLOAT* row);
  virtual void init(SVMINT new_cache_MB,example_set_c* new_examples);
  virtual void overwrite(const SVMINT i, const SVMINT j);
};


class kernel_complete_matrix_c : public kernel_c{
 protected:
  SVMFLOAT* matrix;
  kernel_c* inner_kernel;
  SVMINT calc_index(const SVMINT i, const SVMINT j);
  SVMINT safe_calc_index(const SVMINT i, const SVMINT j);
 public:
  kernel_complete_matrix_c();
  ~kernel_complete_matrix_c();
  void input(std::istream& data_stream);
  void output(std::ostream& data_stream) const;
  virtual void compute_row(const SVMINT i, SVMFLOAT* myrow);
  virtual SVMFLOAT calculate_K(const SVMINT i, const SVMINT j);
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);
  virtual void init(SVMINT new_cache_MB,example_set_c* new_examples);
  virtual void overwrite(const SVMINT i, const SVMINT j);
};


class kernel_aggregation_c : public kernel_c{
 protected:
  SVMINT number_elements;
  kernel_c** elements;
  SVMINT* from;
  SVMINT* to;
  svm_example new_x, new_y;
 public:
  kernel_aggregation_c();
  ~kernel_aggregation_c();
  virtual void init(SVMINT new_cache_MB,example_set_c* new_examples);
  void input(std::istream& data_stream);
  void output(std::ostream& data_stream) const;
  void output_aggregation(std::ostream& data_stream) const;
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);
};


class kernel_prod_aggregation_c : public kernel_aggregation_c{
 public:
  kernel_prod_aggregation_c();
  ~kernel_prod_aggregation_c();
  void output(std::ostream& data_stream) const;
  virtual SVMFLOAT calculate_K(const svm_example x, const svm_example y);
};



// container class

class kernel_container_c{
 protected:
  kernel_c* kernel;
 public:
  friend std::istream& operator >> (std::istream& data_stream, kernel_container_c& the_container);
  friend std::ostream& operator << (std::ostream& data_stream, kernel_container_c& the_container);
  kernel_container_c(){ kernel = 0; };
  ~kernel_container_c();
  kernel_c* get_kernel();
  void clear();
  int is_linear; // dot-kernel?
};


#endif
