#include <coin/CbcModel.hpp>
#include <coin/OsiClpSolverInterface.hpp>
#include <iostream>
#include <cmath>
#include <cassert>


namespace {
namespace local {

// REF [doc] >> ${COIN-OR_HOME}/COIN-OR/doc/cbcuserguide.html#firstexample
// REF [function] >> local::osi_example() in ./coin_or_osi_example()
void cbc_simple_example()
{
	OsiClpSolverInterface solver;

	// Read in example model in MPS file format and assert that it is a clean model.
#if 0
	// Path cover problem.
	//	- RDA data: 2016/04/06, adaptor 1, side 0deg.
	const int numMpsReadErrors = solver.readMps("./data/optimization/path_cover_problem.mps", "");
#else
	const int numMpsReadErrors = solver.readMps("./data/optimization/p0033.mps", "");
#endif
	assert(numMpsReadErrors == 0);

	// Pass the solver with the problem to be solved to CbcModel.
	CbcModel model(solver);

	// Do complete search.
	model.branchAndBound();

	// Print the solution.
	// REF [doc] >> ${COIN-OR_HOME}/COIN-OR/doc/cbcuserguide.html#gettingsolution
	//	Primal column solution: OsiSolverInterface::getColSolution(). It is safer to use CbcModel version, CbcModel::bestSolution().
	//	Dual row solution: OsiSolverInterface::getRowPrice() = CbcModel::getRowPrice().
	//	Primal row solution: OsiSolverInterface::getRowActivity() = CbcModel::getRowActivity().
	//	Dual column solution: OsiSolverInterface::getReducedCost() = CbcModel::gtReducedCost().
	//	Number of rows in model: OsiSolverInterface::getNumRows() = CbcModel::getNumRows(). The number of rows can change due to cuts.
	//	Number of columns in model: OsiSolverInterface::getNumCols() = CbcModel::getNumCols().

	// CbcModel clones the solver so we need to get current copy from the CbcModel.
	const int numColumns = model.solver()->getNumCols();
	const double *solution = model.bestSolution();

	std::cout << "#solutions = " << model.getSolutionCount() << std::endl;
	std::cout << "#saved solutions = " << model.numberSavedSolutions() << std::endl;
	for (int iCol = 0; iCol < numColumns; ++iCol)
	{
		const double &value = solution[iCol];
		if (std::abs(value) > 1.0e-7 && model.solver()->isInteger(iCol))
			std::cout << '\t' << iCol << " has value " << value << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_coin_or {

void cbc_example()
{
	local::cbc_simple_example();
}

}  // namespace my_coin_or
