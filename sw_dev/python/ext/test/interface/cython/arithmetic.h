#if !defined(ARITHMETIC_H)
#define ARITHMETIC_H

namespace arithmetic {

double add(const double lhs, const double rhs);
double sub(const double lhs, const double rhs);
double mul(const double lhs, const double rhs);
double div(const double lhs, const double rhs);

/*
double add(const double lhs, const double rhs)
{
	return lhs + rhs;
}

double sub(const double lhs, const double rhs)
{
	return lhs - rhs;
}

double mul(const double lhs, const double rhs)
{
	return lhs * rhs;
}

double div(const double lhs, const double rhs)
{
	return lhs / rhs;
}
*/

}  // namespace arithmetic

#endif  // ARITHMETIC_H

