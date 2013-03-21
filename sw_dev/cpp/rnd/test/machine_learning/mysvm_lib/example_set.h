#ifndef example_set_h
#define example_set_h 1

#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include "globals.h"


/**
 * Storage class for the examples
 *
 * This class stores the SVM examples and provides simple operations on it
 * @li access to the attributes, classification and lagrangian multipliers
 * @li reading and writing to file 
 * @li scaling
 * Attention! All access to the examples is call-by-reference.
 *
 * @author Stefan Rueping <rueping@ls8.cs.uni-dortmund.de>
 * @version 0.1
 **/


class example_set_c{
 private:
  SVMINT examples_total; // number of examples in the set
  SVMINT capacity; // capacity of the_set
  svm_example* the_set; // Vector of examples, each a array of dim+2 SVMFLOATs (x,y,alpha)
  SVMFLOAT b; // constant of hyperplane (f(x) = wx+b)
  // Expectancy and variance, updated by scale(). Scale-factors may be needed later;
  SVMFLOAT* Exp; // size dim+1, last entry: y
  SVMFLOAT* Var;
  int has_y, has_alphas, has_scale, has_pattern_y; // do y_i and alpha_i and Exp,Var have correct values? Are all y in {-1,1}?
  SVMFLOAT* all_alphas;
  SVMFLOAT* all_ys;
  SVMINT dim; // dimensionality of the examples
  char* filename; // name of file examples were read from
  /** 
   * really do the scaling work
   **/
  void do_scale();
 public:
  example_format my_format;
  void set_filename(char* new_filename); // name of file examples were read from
  char* get_filename(){ return(filename); };
  friend std::istream& operator >> (std::istream& data_stream, example_set_c& examples);
  friend std::ostream& operator << (std::ostream& data_stream, example_set_c& examples);
  /**
   * Constructor. Get Number of examples and dimensionality and set up data structures
   *
   **/
  example_set_c();
  example_set_c(SVMINT new_total, SVMINT new_dim);
  void init(SVMINT new_total, SVMINT new_dim);
  /**
   * Destruktor: delete alle examples
   */
  ~example_set_c();
  /**
   *
   * set the default file format
   *
   **/
  void set_format(example_format new_format);
  /**
   *
   * Set dimension (can be set higher, but not lower)
   *
   */
  void set_dim(SVMINT new_dim);
  SVMINT get_dim();
  /**
   * No of examples
   **/
  SVMINT size();
  /*
   * No. of positive / negative examples
   */
  SVMINT size_pos();
  SVMINT size_neg();
  /**
   * Change the number of the examples. 
   **/
  void resize(SVMINT new_total);
  void compress();
  /**
   * Access functions to the examples.
   **/
  void put_example(const SVMINT pos, const SVMFLOAT* example);
  void put_example(const SVMFLOAT* example); // add one example
  void put_example(const SVMINT pos, const svm_example example);
  void put_example(const svm_example example); // add one example
  svm_example get_example(const SVMINT pos);
  void put_y(const SVMINT pos, const SVMFLOAT y);
  SVMFLOAT get_y(const SVMINT pos); // input y
  SVMFLOAT get_y_var();
  SVMFLOAT unscale_y(const SVMFLOAT scaled_y);
  void put_alpha(const SVMINT pos, const SVMFLOAT alpha);
  SVMFLOAT get_alpha(const SVMINT pos);
  void put_b(const SVMFLOAT new_b);
  SVMFLOAT get_b();
  SVMFLOAT* get_alphas();
  SVMFLOAT* get_ys();
  void put_Exp_Var(SVMFLOAT* newExp, SVMFLOAT* newVar);
  void swap(SVMINT i, SVMINT j);

  /**
   * Are y and alpha initialised?
   *
   * When reading examples to predict or complete model
   **/
  int initialised_y(){ return has_y; };
  int initialised_alpha(){ return has_alphas; };
  int initialised_scale(){ return has_scale; };
  int initialised_pattern_y(){ return has_pattern_y; };

  /**
   * Define alpha or y to be initialized
   *
   */
  void set_initialised_y(){ has_y = 1; };
  void set_initialised_alpha(){ has_alphas = 1; };

  /**
   * scale alphas (alpha -> factor*alpha)
   **/
  void scale_alphas(const SVMFLOAT factor);
  /** 
   * scale the attributes to expectancy 0 and deviation 1
   **/
  void scale();
  void scale(int scale_y);
  /** 
   * scale first scaledim attributes to x[i] = (x[i] - const[i])/factor[i]
   **/
  void scale(SVMFLOAT *theconst, SVMFLOAT *thefactor,SVMINT scaledim);
  /**
   * get expectancy
   */ 
  SVMFLOAT* get_exp(){ return Exp; };
  /**
   * get variance
   */
  SVMFLOAT* get_var(){ return Var; };
  /**
   * clear all data
   **/
  void clear();
  /**
   * clear alpha values
   **/
  void clear_alpha();
  /**
   * Sum of all alphas, should be zero. (for debugging)
   **/
  SVMFLOAT sum();

  /**
   * permute the examples
   */
  void permute();


  void output_ys(std::ostream& data_stream) const;
};        

std::ostream& operator<< (std::ostream& data_stream, example_set_c& examples);
std::istream& operator>> (std::istream& data_stream, example_set_c& examples);

#endif
