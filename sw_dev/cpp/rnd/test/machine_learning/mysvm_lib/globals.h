#ifndef globals_h
#define globals_h 1


//#define windows 1

#define use_time 1
// uncomment the following line if you get problems with sys/times.h when compiling:
#undef use_time

#ifdef windows
#undef use_time
#define isnan _isnan
#endif


#ifdef use_time
#include <sys/times.h>
#endif

#include <string.h>
#include <math.h>
#include <iostream>


/**
 * Global declarations
 *
 * Declares
 * @li Variable types
 *
 * @author Stefan Rueping <rueping@ls8.cs.uni-dortmund.de>
 * @version 0.1
 **/


// variable types
typedef long SVMINT;
typedef double SVMFLOAT;

// constants
#define MAXSVMINT 2147483647
#define infinity 1e20
#define MAXCHAR 10000
#define PI 3.1415926535


typedef struct svm_attrib{
  SVMFLOAT att;
  SVMINT index;
} SVM_ATTRIB;

typedef struct svm_example{
  SVMINT length;
  svm_attrib* example;
  SVMFLOAT y;
  SVMFLOAT alpha;
} SVM_EXAMPLE;

typedef struct svm_result {
  SVMFLOAT VCdim;
  SVMFLOAT loss;
  SVMFLOAT accuracy;
  SVMFLOAT precision;
  SVMFLOAT recall;
  SVMFLOAT MAE;
  SVMFLOAT MSE;
  // for the asymmetrical case
  SVMFLOAT loss_pos;
  SVMFLOAT loss_neg;
  // loo predictors
  SVMFLOAT pred_loss;
  SVMFLOAT pred_accuracy;
  SVMFLOAT pred_precision;
  SVMFLOAT pred_recall;
  // count of the support vectors
  SVMINT number_svs;
  SVMINT number_bsv;
} SVM_RESULT;

typedef struct quadratic_program {
  SVMINT    n;   /* number of variables */
  SVMINT    m;   /* number of linear equality constraints */
  SVMFLOAT* c;   
  SVMFLOAT* H;   /* c' * x + 1/2 x' * H * x -> min */
  SVMFLOAT* A;   
  SVMFLOAT* b;   /* A * x = b */
  SVMFLOAT* l;   
  SVMFLOAT* u;   /* l <= x <= u */
} QP;


class example_format{
 public:
  // format of examples file
  int sparse;
  int where_x;
  int where_y ;
  int where_alpha;
  char delimiter;
};

std::ostream& operator<< (std::ostream& data_stream, example_format& format);


// exceptions 
class general_exception{
 public:
  char* error_msg;
  general_exception();
  general_exception(char* the_error);
};

class read_exception : public general_exception{
 public:
  read_exception();
  read_exception(char* the_error);
};

class no_number_exception : public general_exception{};

class input_exception : public general_exception{};

// little helpers
#define abs(x) ((x) >= 0 ? (x) : -(x))


SVMFLOAT x_i(const svm_example x, const SVMINT i);

SVMFLOAT string2svmfloat(char* s);

long get_time();

#endif


