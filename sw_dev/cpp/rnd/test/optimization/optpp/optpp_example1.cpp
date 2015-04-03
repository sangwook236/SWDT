#include <NLP.h>
#include <NLF.h>
#include <OptQNewton.h>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

// minimize 100(x_2 - x_{1}^2)^2 + (1 - x_1)^2.

void init_rosen(const int ndim, NEWMAT::ColumnVector &x)
{
	if (2 != ndim)
		throw std::runtime_error("incorrect vector dimension : ndim == 2");

	// ColumnVectors are indexed from 1, and they use parentheses around the index.
	x(1) = -1.2;
	x(2) =  1.0;
}

void rosen(const int ndim, const NEWMAT::ColumnVector &x, double &fx, int &result)
{
	if (2 != ndim)
		throw std::runtime_error("incorrect vector dimension : ndim == 2");

	const double x1 = x(1);
	const double x2 = x(2);
	const double f1 = (x2 - x1 * x1);
	const double f2 = 1. - x1;

	fx = 100.* f1*f1 + f2*f2;
	result = OPTPP::NLPFunction;
}

}  // namespace local
}  // unnamed namespace

namespace my_optpp {

// Example 1: Unconstrained Quasi-Newton Without Derivatives
//  [ref] https://software.sandia.gov/opt++/opt++2.4_doc/html/example1.html
void example1(int argc, char *argv[])
{
	const int ndim = 2;

	OPTPP::FDNLF1 nlp(ndim, local::rosen, local::init_rosen);
	OPTPP::OptQNewton objfcn(&nlp);

	objfcn.setSearchStrategy(OPTPP::TrustRegion);
	objfcn.setMaxFeval(200);
	objfcn.setFcnTol(1.e-4);

	// The "0" in the second argument says to create a new file.
	// A "1" would signify appending to an existing file.
	if (!objfcn.setOutputFile("./data/optimization/optpp/example1.out", 0))
		std::cerr << "main: output file open failed" << std::endl;

	objfcn.optimize();

	objfcn.printStatus("Solution from quasi-newton");
	objfcn.cleanup();
}

}  // namespace my_optpp
