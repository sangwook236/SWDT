#include "globals.h"
#include <fstream>


/*
 *
 * solve  c' * x + 1/2 x' * H * x -> min
 * w.r.t. A * x = b
 *        l <= x <= u
 *
 */
 

class smo_c {
 private:
  quadratic_program* qp;
  SVMFLOAT* x;
  SVMFLOAT* sum;
  SVMINT n;
  SVMFLOAT lambda_eq;
  SVMFLOAT lambda_nu;

  SVMFLOAT is_zero;
  SVMFLOAT max_allowed_error;
  SVMINT max_iteration;

  SVMFLOAT x2tox1(const SVMFLOAT x2, const int id, 
		  const SVMFLOAT A1, const SVMFLOAT b);

  SVMFLOAT x1tox2(const SVMFLOAT x1, const int id, 
		  const SVMFLOAT A2, const SVMFLOAT b);

  void simple_solve(SVMFLOAT* x1, SVMFLOAT* x2,
		    const SVMFLOAT H1, const SVMFLOAT H2,
		    const SVMFLOAT c0, 
		    const SVMFLOAT c1, const SVMFLOAT c2,
		    const SVMFLOAT A1, const SVMFLOAT A2,
		    const SVMFLOAT l1, const SVMFLOAT l2,
		    const SVMFLOAT u1, const SVMFLOAT u2);

  int minimize_ij(const SVMINT i, const SVMINT j);
  int minimize_i(const SVMINT i);

  void calc_lambda_eq();
  void calc_lambda_nu();

  void set_qp(quadratic_program* the_qp);

 public:
  smo_c();
  smo_c(const SVMFLOAT new_is_zero, 
	const SVMFLOAT new_max_allowed_error, 
	const SVMINT new_max_iteration);
  void set_max_allowed_error(SVMFLOAT new_max_allowed_error);
  void init(const SVMFLOAT new_is_zero, 
       const SVMFLOAT new_max_allowed_error, 
       const SVMINT new_max_iteration);
  int smo_solve(quadratic_program* the_qp,SVMFLOAT* the_x);
  int smo_solve_single(quadratic_program* the_qp,SVMFLOAT* the_x);
  int smo_solve_const_sum(quadratic_program* the_qp,SVMFLOAT* the_x);
  SVMFLOAT get_lambda_eq();
  SVMFLOAT get_lambda_nu();
};
