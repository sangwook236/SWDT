#ifndef parameters_h
#define parameters_h 1

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "globals.h"

/**
 * Class for all SVM-parameters
 * @li read and write access to all parameters
 * @li stream input and output
 *
 * @author Stefan Rueping <rueping@ls8.cs.uni-dortmund.de>
 * @version 0.1
 **/



class parameters_c{
  // neg means "f(x) < y-eps" (<=> "*")
  // machine and kernel type
    // capacity
 public:
  // input and output functions
  friend std::istream& operator >> (std::istream& data_stream, parameters_c& the_parameters);
  friend std::ostream& operator << (std::ostream& data_stream, parameters_c& the_parameters);

  // default methods
  parameters_c();
  void clear();

  // capacity constraint
  SVMFLOAT realC; 

  // loss function
  SVMFLOAT Lpos, Lneg;
  SVMFLOAT epsilon_pos, epsilon_neg;
  int quadraticLossPos, quadraticLossNeg;
  int balance_cost;

  // default parameters for examples
  int do_scale; // scale examples
  int do_scale_y; // scale y-values
  example_format default_example_format;

  // type of SVM
  int is_pattern; // set Lpos=0 for y>0 and Lneg=0 for y<0
  int is_linear; // kernel=dot, use folding
  int is_distribution;
  int biased; // biased hyperplane (w*x+b) or unbiased (w*x)?

  // do cross validation?
  SVMINT cross_validation; // cross-validate on training set
  int cv_window; // do cross-validation by means of a sliding window
  int cv_inorder; // do cross-validation in given order of examples

  // parameters for search of C
  char search_c;
  SVMINT search_stop;
  SVMFLOAT c_min; // search for C to minimize loss
  SVMFLOAT c_max;
  SVMFLOAT c_delta;

  // numerical optimization parameters
  SVMFLOAT is_zero;  // when is a lagrangian multiplier considered 0
  SVMFLOAT nu; // nu-SVM
  int is_nu;

  SVMINT max_iterations;
  SVMINT working_set_size;
  SVMINT shrink_const;
  SVMFLOAT descend;  // make at least this much descend on WS
  SVMFLOAT convergence_epsilon;
  SVMINT kernel_cache;

  int use_min_prediction;
  SVMFLOAT min_prediction; // let pred =  max(min_prediction,f(x))

  /**
   * Verbosity (higher level includes smaller):
   * 0 : only critical errors
   * 1 : information about success of algorithm
   * 2 : small summary about training and test
   * 3 : larger summary about training
   * 4 : information about each iteration
   * 5 : flood
   */
  int verbosity;
  int print_w; // print whole hyperplane?
  int loo_estim; // print loo estim?

  SVMFLOAT get_Cpos(){ return(Lpos*realC); };
  SVMFLOAT get_Cneg(){ return(Lneg*realC); };
};

std::istream& operator >> (std::istream& data_stream, parameters_c& the_parameters);
std::ostream& operator << (std::ostream& data_stream, parameters_c& the_parameters);

#endif
