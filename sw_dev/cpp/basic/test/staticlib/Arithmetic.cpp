#include "staticlib/Arithmetic.h"


const int TWO = 2;

int DOUBLE(const int val)
{
	return 2 * val;
}

/*static*/ const int Arithmetic::ONE = 1;

/*static*/ int Arithmetic::add(const int lhs, const int rhs)
{
	return lhs + rhs;
}

/*static*/ int Arithmetic::subtract(const int lhs, const int rhs)
{
	return lhs - rhs;
}

int Arithmetic::add(const int rhs) const
{
	return lhs_ + rhs;
}

int Arithmetic::subtract(const int rhs) const
{
	return lhs_ - rhs;
}

void Arithmetic::multiply(const int rhs)
{
	lhs_ *= rhs;
}
