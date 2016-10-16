#include <NLP.h>
#include <NLF.h>
#include <BoundConstraint.h>
#include <NonLinearInequality.h>
#include <ompoundConstraint.h>
#include <OptNIPS.h>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

// Minimize (x_1 - x_2)^2 + (1/9)(x_1 + x_2 - 10)^2 + (x_3 - 5)^2
// subject to x_1^2 + x_2^2 + x_3^2 <= 48
//            -4.5 <= x_1 <= 4.5
//            -4.5 <= x_2 <= 4.5
//            -5.0 <= x_3 <= 5.0

void init_hs65(const int ndim, NEWMAT::ColumnVector &x)
{
	if (3 != ndim)
		throw std::runtime_error("incorrect vector dimension : ndim == 3");

	double factor = 0.0;

	// ColumnVectors are indexed from 1, and they use parentheses around the index.
	x(1) = -5.0 - (factor - 1)*8.6505;
	x(2) =  5.0 + (factor - 1)*1.3495;
	x(3) =  0.0 - (factor - 1)*4.6204;
}

void hs65(const int mode, const int ndim, const NEWMAT::ColumnVector &x, double &fx, NEWMAT::ColumnVector &gx, NEWMAT::SymmetricMatrix &Hx, int &result)
{
	if (3 != ndim)
		throw std::runtime_error("incorrect vector dimension : ndim == 3");

	const double x1 = x(1);
	const double x2 = x(2);
	const double x3 = x(3);
	const double f1 = x1 - x2;
	const double f2 = x1 + x2 - 10.0;
	const double f3 = x3 - 5.0;

	if (NLPFunction & mode)
	{
		fx = f1*f1 + (f2*f2)/9.0 + f3*f3;
		result = OPTPP::NLPFunction;
	}

	if (NLPGradient & mode)
	{
		gx(1) =  2*f1 + (2.0/9.0)*f2;
		gx(2) = -2*f1 + (2.0/9.0)*f2;
		gx(3) =  2*f3;
		result = OPTPP::NLPGradient;
	}

	// The various Matrix objects have two indices, are indexed from 1, and they use parentheses around the index.
	if (mode & OPTPP::NLPHessian)
	{
		Hx(1,1) =  2 + (2.0/9.0);

		Hx(2,1) = -2 + (2.0/9.0);
		Hx(2,2) =  2 + (2.0/9.0);

		Hx(3,1) = 0.0;
		Hx(3,2) = 0.0;
		Hx(3,3) = 2.0;
		result = OPTPP::NLPHessian;
	}
}

void ineq_hs65(const int mode, const int ndim, const NEWMAT::ColumnVector &x, NEWMAT::ColumnVector &cx, NEWMAT::Matrix &cgx, OPTPP::OptppArray<NEWMAT::SymmetricMatrix> &cHx, int &result)
{
	// Hock and Schittkowski's Problem 65 

	if (ndim != 3)
		throw std::runtime_error("incorrect vector dimension : ndim == 3");

	const double x1 = x(1);
	const double x2 = x(2);
	const double x3 = x(3);
	const double f1 = x1;
	const double f2 = x2;
	const double f3 = x3;

	if (mode & OPTPP::NLPFunction)
	{
		cx(1) = 48 - f1*f1 - f2*f2 - f3*f3;
		result = OPTPP::NLPFunction;
	}

	if (mode & OPTPP::NLPGradient)
	{
		cgx(1,1) = -2*x1;
		cgx(2,1) = -2*x2;
		cgx(3,1) = -2*x3;
		result = OPTPP::NLPGradient;
	}

	if (mode & OPTPP::NLPHessian)
	{
		NEWMAT::SymmetricMatrix Htmp(ndim);
		Htmp(1,1) = -2;
		Htmp(1,2) = 0.0;
		Htmp(1,3) = 0.0;
		Htmp(2,1) = 0.0;
		Htmp(2,2) = -2;
		Htmp(2,3) = 0.0;
		Htmp(3,1) = 0.0;
		Htmp(3,2) = 0.0;
		Htmp(3,3) = -2;

		cHx[0] = Htmp;
		result = OPTPP::NLPHessian;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_optpp {

// Example 2: Nonlinear Interior-Point Method With General Constraints
//  [ref] https://software.sandia.gov/opt++/opt++2.4_doc/html/example2.html
void example2(int argc, char *argv[])
{
	const int ndim = 3;
	NEWMAT::ColumnVector lower(ndim), upper(ndim); 

	// Here is one way to assign values to a ColumnVector.
	lower << -4.5 << -4.5 << -5.0;
	upper <<  4.5 <<  4.5 <<  5.0 ;

	OPTPP::Constraint c1 = new OPTPP::BoundConstraint(ndim, lower, upper);

	OPTPP::NLP *chs65 = new OPTPP::NLP(new OPTPP::NLF2(ndim, 1, ineq_hs65, init_hs65));
	OPTPP::Constraint nleqn = new OPTPP::NonLinearInequality(chs65);

	OPTPP::CompoundConstraint* constraints = new OPTPP::CompoundConstraint(nleqn, c1);

	OPTPP::NLF2 nips(ndim, hs65, init_hs65, constraints);
	OPTPP::OptNIPS objfcn(&nips);

	// The "0" in the second argument says to create a new file.
	// A "1" would signify appending to an existing file.
	objfcn.setOutputFile("./data/optimization/optpp/example2.out", 0);
	objfcn.setFcnTol(1.0e-06);
	objfcn.setMaxIter(150);
	objfcn.setMeritFcn(ArgaezTapia);

	objfcn.optimize();

	objfcn.printStatus("Solution from nips");
	objfcn.cleanup();
}

}  // namespace my_optpp
