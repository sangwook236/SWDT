#include "sharedlib/Trigonometric.h"
#include <cmath>



const double E = std::exp(1.0);
double SQRT(const double val)
{
	return std::sqrt(val);
}

/*static*/ const double Trigonometric::PI = 4.0 * std::atan(1.0);

/*static*/ double Trigonometric::sin(const double val)
{
	return std::sin(val);
}

double Trigonometric::cos() const
{
	return std::cos(val_);
}

void Trigonometric::tan()
{
	val_ = std::tan(val_);
}
