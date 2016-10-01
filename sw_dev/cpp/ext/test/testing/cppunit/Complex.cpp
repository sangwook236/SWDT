//#include "stdafx.h"
#include "Complex.h"


namespace constant {

const double EPS = 1.0e-5;
const double _2_PI = 8.0 * atan(1.0);

}

namespace util {

bool is_zero(double x, double tol/* = constant::EPS*/)
{  return -tol <= x && x <= tol;  }

}
