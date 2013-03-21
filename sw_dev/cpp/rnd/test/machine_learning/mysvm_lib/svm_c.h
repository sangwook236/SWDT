#ifndef svm_c_h
#define svm_c_h 1

#include <stdlib.h>
#include <math.h>
#include "parameters.h"
#include "kernel.h"
#include "example_set.h"
#include "globals.h"
#include "smo.h"

/**
 * base class for SVMs
 *
 * @author Stefan Rueping <rueping@ls8.cs.uni-dortmund.de>
 * @version 0.1
 **/


class svm_c {
 protected:
  smo_c smo;
  SVMFLOAT* all_alphas;
  SVMFLOAT* all_ys;

  SVMINT examples_total;
  SVMINT working_set_size;
  SVMINT shrink_const; // a lagrangian multiplier is fixed if it is at bound more than this number of times
  SVMINT to_shrink;
  SVMFLOAT is_zero;
  SVMFLOAT Cpos, Cneg; // C and C*
  SVMFLOAT epsilon_pos, epsilon_neg;
  SVMFLOAT lambda_eq; // lagrangian multiplier of equality contraint
  SVMFLOAT lambda_WS; // lagrangian multiplier on working set
  SVMINT target_count; // how long no descend in WS?
  SVMFLOAT sum_alpha;
  int biased; // biased hyperplane (w*x+b) or unbiased (w*x)

  example_set_c* examples;
  example_set_c* test_set;
  kernel_c* kernel;
  parameters_c* parameters;

  // ex_i, op_i and ws_i are used as indices.
  // ex_i: index in examples, ex_i = index[op_i], ws_i
  // op_i: index of sorted examples
  // ws_i: index in working set, i.e. index in op_i if fixed values are omitted

  // The following arrays range over the example set
  SVMFLOAT* sum; // sum_i = sum_{j=1}^n (alpha_j^*-alpha_j) K(i,j)
                 // i=ex_i= 1..examples_total
  SVMINT* which_alpha; // -1 if alpha_i in WS, 1 if alpha_i^*, 0 otherwise
                       // i.e. which_alpha[i]*alpha[i] = examples->get_alpha(i)
                       // i=ex_i= 1..examples_total
  SVMINT* at_bound; // how many it. has var been at bound
  quadratic_program qp;

  SVMFLOAT* primal; // workspace for primal variables

  SVMINT* working_set;
  SVMFLOAT* working_set_values;
  // LOQO-parameters:
  SVMFLOAT init_margin;
  SVMFLOAT init_bound;
  SVMFLOAT sigfig_max;

  int init_counter_max;
  int is_pattern;
  SVMFLOAT convergence_epsilon;
  SVMFLOAT feasible_epsilon;

  /*
   * check time usage
   */
  long time_init;
  long time_optimize;
  long time_convergence;
  long time_update;
  long time_calc;
  long time_all;

  virtual int feasible(const SVMINT i, SVMFLOAT* the_nabla, SVMFLOAT* the_lambda, int* atbound); // feasible variable?
  virtual SVMFLOAT nabla(const SVMINT i); // gradient
  virtual SVMFLOAT lambda(const SVMINT i); // lagrangian multiplier
  virtual int is_alpha_neg(const SVMINT i); // alpha or alpha* ?

  virtual void calculate_working_set();
  void minheap_heapify(const SVMINT start, const SVMINT size);
  void maxheap_heapify(const SVMINT start, const SVMINT size);

  SVMINT minheap_add(SVMINT size, const SVMINT element, const SVMFLOAT value);
  SVMINT maxheap_add(SVMINT size, const SVMINT element, const SVMFLOAT value);

  /**
   * Call actual optimizer
   **/
  virtual void optimize();
  void exit_optimizer();
  virtual int convergence();
  virtual void init_working_set();
  virtual void update_working_set();
  void put_optimizer_values();
  virtual void shrink();
  virtual void reset_shrinked();
  virtual void project_to_constraint();

  SVMFLOAT svm_c::avg_norm2(); // avg_examples(K(x,x))
  SVMFLOAT loss(SVMFLOAT prediction, SVMFLOAT value); // the actual loss-function
  SVMFLOAT loss(SVMINT i); // loss of example i
  SVMFLOAT predict(svm_example example); // calculate (regression-)prediction for one example
  SVMFLOAT predict(SVMINT i); // predict example i
  virtual void print_special_statistics();  // print statistics related to some subtype of SVM
 protected:
  virtual void init_optimizer();
 public:
  svm_c();
  /**
   * initialise kernel and parameters
   */
  virtual void init(kernel_c* new_kernel, parameters_c* new_parameters);
  /**
   * Train the SVM 
   **/
  svm_result train(example_set_c* training_examples);
  /**
   * Test svm on test set
   */
  svm_result test(example_set_c* training_examples, int verbose);
  /*
   * Predict values for exmaples in test set
   **/
  void predict(example_set_c* training_examples);
  /**
   * Init examples for testing or predicting
   **/
  void set_svs(example_set_c* training_examples);
  /*
   * print information about test set
   **/
  svm_result print_statistics();
};


class svm_pattern_c : public svm_c {
 public:
  svm_pattern_c() : svm_c() {};
  virtual int feasible(const SVMINT i, SVMFLOAT* the_nabla, SVMFLOAT* the_lambda, int* atbound); // feasible variable?
  virtual SVMFLOAT nabla(const SVMINT i); // gradient
  virtual SVMFLOAT lambda(const SVMINT i); // lagrangian multiplier
  virtual int is_alpha_neg(const SVMINT i); // alpha or alpha* ?
 protected:
  virtual void init_optimizer();
};


class svm_regression_c : public svm_c {
 public:
  svm_regression_c() : svm_c() {};
  virtual int feasible(const SVMINT i, SVMFLOAT* the_nabla, SVMFLOAT* the_lambda, int* atbound); // feasible variable?
  virtual SVMFLOAT nabla(const SVMINT i); // gradient
  virtual SVMFLOAT lambda(const SVMINT i); // lagrangian multiplier
  virtual int is_alpha_neg(const SVMINT i); // alpha or alpha* ?
};


#endif
