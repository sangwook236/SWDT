#ifndef svm_nu_h
#define svm_nu_h 1

#include "svm_c.h"

/**
 * base class for nu SVMs
 *
 * @author Stefan Rueping <rueping@ls8.cs.uni-dortmund.de>
 * @version 0.1
 *
 **/


class svm_nu_regression_c : public svm_c{
 protected:
  SVMFLOAT lambda_nu;
  SVMFLOAT lambda_nu_WS;
  SVMFLOAT sum_alpha_nu;
  SVMFLOAT nu;
  virtual void reset_shrinked();
  virtual void init(kernel_c* new_kernel, parameters_c* new_parameters);
  virtual void init_optimizer();
  virtual int is_alpha_neg(const SVMINT i);
  virtual SVMFLOAT lambda(const SVMINT i);
  virtual int feasible(const SVMINT i);
  virtual SVMFLOAT nabla(const SVMINT i);
  virtual void project_to_constraint();
  virtual int convergence();
  virtual void init_working_set();
  virtual void shrink();
  virtual void optimize();
  virtual void print_special_statistics();
 public:
  svm_nu_regression_c() : svm_c() { lambda_nu = 0; };
};


class svm_nu_pattern_c : public svm_nu_regression_c{
 protected:
  virtual SVMFLOAT nabla(const SVMINT i);
  virtual void init(kernel_c* new_kernel, parameters_c* new_parameters);
  virtual void init_optimizer();
  virtual void update_working_set();
  virtual void init_working_set();
  virtual void print_special_statistics();
 public:
  svm_nu_pattern_c() : svm_nu_regression_c() {};
};


class svm_distribution_c : public svm_pattern_c{
 protected:
  SVMFLOAT nu;
  virtual int is_alpha_neg(const SVMINT i);
  virtual SVMFLOAT nabla(const SVMINT i);
  virtual SVMFLOAT lambda(const SVMINT i);
  virtual int feasible(const SVMINT i);
  virtual int feasible(const SVMINT i, SVMFLOAT* the_nabla, SVMFLOAT* the_lambda, int* atbound);
  virtual void init(kernel_c* new_kernel, parameters_c* new_parameters);
  virtual void init_optimizer();
  virtual void project_to_constraint();
  virtual int convergence();
  virtual void init_working_set();
  virtual void print_special_statistics();
 public:
  svm_distribution_c() : svm_pattern_c() {};
};
#endif
