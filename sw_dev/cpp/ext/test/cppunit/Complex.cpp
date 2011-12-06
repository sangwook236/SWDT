#include "Complex.h"


#if defined(_MSC_VER) && defined(_DEBUG)
#define VC_EXTRALEAN  //  Exclude rarely-used stuff from Windows headers
//#include <afx.h>
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif


namespace constant {

const double EPS = 1.0e-5;
const double _2_PI = 8.0 * atan(1.0);

}

namespace util {

bool is_zero(double x, double tol/* = constant::EPS*/)
{  return -tol <= x && x <= tol;  }

}
