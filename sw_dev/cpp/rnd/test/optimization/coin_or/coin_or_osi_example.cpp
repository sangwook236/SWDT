#include <coin/OsiClpSolverInterface.hpp>
#include <coin/OsiGlpkSolverInterface.hpp>
#include <coin/CoinPackedMatrix.hpp>
#include <coin/CoinPackedVector.hpp>
#include <iostream>
#include <string>
#include <memory>
#include <cassert>


namespace {
namespace local {

// REF [doc] >> "5.4 parameters.cpp" in "${COIN-OR_HOME}/COIN-OR/doc/A Gentle Introduction to Optimization Solver Interface.pdf"
void osi_basic_example()
{
	// Create a problem pointer. We use the base class here.
	// When we instantiate the object, we need a specific derived class.
	std::shared_ptr<OsiSolverInterface> si(new OsiClpSolverInterface);
	//std::shared_ptr<OsiSolverInterface> si(new OsiGlpkSolverInterface);

	// Read in an mps file. This one's from the MIPLIB library.
	si->readMps("./data/optimization/p0033");

#if 0
	// Demonstrate some problem and solution query methods and also demonstrate some parameter setting.
	{
		// Display some information about the instance.
		const int nrows = si->getNumRows();
		const int ncols = si->getNumCols();
		const int nelem = si->getNumElements();
		std::cout << "This problem has " << nrows << " rows, " << ncols << " columns, and " << nelem << " nonzeros." << std::endl;

		const double const * upper_bounds = si->getColUpper();
		std::cout << "The upper bound on the first column is " << upper_bounds[0] << std::endl;

		// All the information about the instance is available with similar methods.
		// Before solving, indicate some parameters.
		si->setIntParam(OsiMaxNumIteration, 10);
		si->setDblParam(OsiPrimalTolerance, 0.001);

		// Can also read parameters.
		std::string solver;
		si->getStrParam(OsiSolverName, solver);
		std::cout << "About to solve with: " << solver << std::endl;
	}
#endif

#if 0
	// Access solver-specific functions.
	//	REF [doc] >> "5.6 specific.cpp" in "${COIN-OR_HOME}/COIN-OR/doc/A Gentle Introduction to Optimization Solver Interface.pdf"
	{
		// The next few lines are solver-dependent!
		ClpSimplex * clpPointer = std::dynamic_pointer_cast<OsiClpSolverInterface>(si)->getModelPtr();
		clpPointer->setLogLevel(0);
		//clpPointer->setMaximumIterations(10);

		// Could tell Clp many other things.
	}
#endif

	// Solve the (relaxation of the) problem.
	si->initialSolve();

	// Check the solution.
	if (si->isProvenOptimal())
	{
		std::cout << "Found optimal solution!" << std::endl;
		std::cout << "Objective value is " << si->getObjValue() << std::endl;

		// Examine solution.
		const int n = si->getNumCols();
		const double *solution = si->getColSolution();

		std::cout << "Solution: ";
		for (int i = 0; i < n; ++i)
			std::cout << solution[i] << " ";
		std::cout << std::endl;
		std::cout << "It took " << si->getIterationCount() << " iterations" << " to solve." << std::endl;
	}
	else
	{
		std::cout << "Didn't find optimal solution." << std::endl;

		// Check other status functions. What happened?
		if (si->isProvenPrimalInfeasible())
			std::cout << "Problem is proven to be infeasible." << std::endl;
		if (si->isProvenDualInfeasible())
			std::cout << "Problem is proven dual infeasible." << std::endl;
		if (si->isIterationLimitReached())
			std::cout << "Reached iteration limit." << std::endl;
	}
}

// REF [doc] >> "5.5 build.cpp" in "${COIN-OR_HOME}/COIN-OR/doc/A Gentle Introduction to Optimization Solver Interface.pdf"
void osi_build_example()
{
	// Create a problem pointer. We use the base class here.
	// When we instantiate the object, we need a specific derived class.
	std::shared_ptr<OsiSolverInterface> si(new OsiClpSolverInterface);

	// Build our own instance from scratch.
	{
		/*
		 * This section adapted from Matt Galati¡¯s example on the COIN-OR Tutorial website.
		 *
		 * Problem from Bertsimas, Tsitsiklis page 21.
		 *
		 * optimal solution: x* = (1,1).
		 *
		 * minimize -1 x0 - 1 x1
		 * s.t	1 x0 + 2 x1 <= 3
		 *		2 x0 + 1 x1 <= 3
		 *		x0 >= 0
		 *		x1 >= 0
		 */
		const int n_cols = 2;
		double * objective = new double [n_cols];  // The objective coefficients.
		double * col_lb = new double [n_cols];  // The column lower bounds.
		double * col_ub = new double [n_cols];  // The column upper bounds.

		// Define the objective coefficients.
		//	minimize -1 x0 - 1 x1
		objective[0] = -1.0;
		objective[1] = -1.0;

		// Define the variable lower/upper bounds.
		//	x0 >= 0 ==> 0 <= x0 <= infinity
		//	x1 >= 0 ==> 0 <= x1 <= infinity
		col_lb[0] = 0.0;
		col_lb[1] = 0.0;
		col_ub[0] = si->getInfinity();
		col_ub[1] = si->getInfinity();
		const int n_rows = 2;
		double * row_lb = new double [n_rows];  // The row lower bounds.
		double * row_ub = new double [n_rows];  // The row upper bounds.

		// Define the constraint matrix.
		CoinPackedMatrix * matrix = new CoinPackedMatrix(false, 0, 0);
		matrix->setDimensions(0, n_cols);

		//	1 x0 + 2 x1 <= 3 => -infinity <= 1 x0 + 2 x2 <= 3
		CoinPackedVector row1;
		row1.insert(0, 1.0);
		row1.insert(1, 2.0);
		row_lb[0] = -1.0 * si->getInfinity();
		row_ub[0] = 3.0;
		matrix->appendRow(row1);

		//	2 x0 + 1 x1 <= 3 => -infinity <= 2 x0 + 1 x1 <= 3
		CoinPackedVector row2;
		row2.insert(0, 2.0);
		row2.insert(1, 1.0);
		row_lb[1] = -1.0 * si->getInfinity();
		row_ub[1] = 3.0;
		matrix->appendRow(row2);

		// Load the problem to OSI.
		si->loadProblem(*matrix, col_lb, col_ub, objective, row_lb, row_ub);

		// Write the MPS file to a file called example.mps.
		si->writeMps("./data/optimization/coin_or/example");
	}

	// Solve the (relaxation of the) problem.
	si->initialSolve();

	// Check the solution.
	if (si->isProvenOptimal())
	{
		std::cout << "Found optimal solution!" << std::endl;
		std::cout << "Objective value is " << si->getObjValue() << std::endl;

		// Examine solution.
		const int n = si->getNumCols();
		const double *solution = si->getColSolution();

		std::cout << "Solution: ";
		for (int i = 0; i < n; ++i)
			std::cout << solution[i] << " ";
		std::cout << std::endl;
		std::cout << "It took " << si->getIterationCount() << " iterations" << " to solve." << std::endl;
	}
	else
	{
		std::cout << "Didn't find optimal solution." << std::endl;

		// Check other status functions. What happened?
		if (si->isProvenPrimalInfeasible())
			std::cout << "Problem is proven to be infeasible." << std::endl;
		if (si->isProvenDualInfeasible())
			std::cout << "Problem is proven dual infeasible." << std::endl;
		if (si->isIterationLimitReached())
			std::cout << "Reached iteration limit." << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_coin_or {

void osi_example()
{
	//local::osi_basic_example();

	// Build the instance internally with sparse matrix object.
	local::osi_build_example();
}

}  // namespace my_coin_or
