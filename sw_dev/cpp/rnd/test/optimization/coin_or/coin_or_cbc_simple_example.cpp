#include <coin/CbcModel.hpp>
#include <coin/OsiClpSolverInterface.hpp>
#include <iostream>
#include <cmath>
#include <cassert>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_coin_or {

// REF [doc] >> ${COIN-OR_HOME}/COIN-OR/doc/cbcuserguide.html#firstexample
void cbc_simple_example()
{
	OsiClpSolverInterface solver1;

	// Read in example model in MPS file format and assert that it is a clean model.
	const int numMpsReadErrors = solver1.readMps("./data/optimization/p0033.mps", "");
	assert(numMpsReadErrors == 0);

	// Pass the solver with the problem to be solved to CbcModel.
	CbcModel model(solver1);

	// Do complete search.
	model.branchAndBound();

	// Print the solution.
	// CbcModel clones the solver so we need to get current copy from the CbcModel.
	const int numberColumns = model.solver()->getNumCols();

	const double *solution = model.bestSolution();

	for (int iColumn = 0; iColumn < numberColumns; ++iColumn)
	{
		const double value = solution[iColumn];
		if (fabs(value) > 1.0e-7 && model.solver()->isInteger(iColumn))
			std::cout << iColumn << " has value " << value << std::endl;
	}
}

}  // namespace my_coin_or
