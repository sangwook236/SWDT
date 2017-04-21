//#include "stdafx.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#define GLOG_NO_ABBREVIATED_SEVERITIES 1
#include <glog/logging.h>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <ctime>


int main(int argc, char *argv[])
{
	int glpk_main(int argc, char *argv[]);

	int ceres_solver_main(int argc, char *argv[]);
	int levmar_main(int argc, char *argv[]);

	int nlopt_main(int argc, char *argv[]);
	int optpp_main(int argc, char *argv[]);

	int coin_or_main(int argc, char *argv[]);
	int scip_main(int argc, char *argv[]);

	int retval = EXIT_SUCCESS;
	try
	{
		std::srand((unsigned int)std::time(NULL));
        google::InitGoogleLogging(argv[0]);

        std::cout << "GNU Linear Programming Kit (GLPK) library ---------------------------" << std::endl;
		//	- Primal and dual simplex method.
		//	- Primal-dual interior-point method.
		//	- Branch-and-cut method.
		//	- Stand-alone LP/MIP solver.
		//	- Graph and Network.
		//	- CNF Satisfiability (CNF-SAT) problem.
		//	- Mathematical programming language.
		//		GNU MathProg (GMPL), MPS format, CPLEX LP format.
		retval = glpk_main(argc, argv);

        std::cout << "\nCeres Solver --------------------------------------------------------" << std::endl;
        //  - Non-linear least squares.
        //  - General unconstrained minimization.
        //      Curve fitting.
        //      Robust curve fitting.
        //      Bundle adjustment.
		//retval = ceres_solver_main(argc, argv);

		std::cout << "\nLevenberg-Marquardt (LM) algorithm ----------------------------------" << std::endl;
		//retval = levmar_main(argc, argv);

        std::cout << "\nNLopt library -------------------------------------------------------" << std::endl;
		//retval = nlopt_main(argc, argv);

        std::cout << "\nOPT++ library -------------------------------------------------------" << std::endl;
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
		//retval = optpp_main(argc, argv);
#else
		std::cout << "\tThis library can be used in unix-like systems" << std::endl;
#endif

		std::cout << "\nComputational Infrastructure for Operations Research (COIN-OR) ------" << std::endl;
		//	- Continuous linear programming.
		//		CLP, DyLP.
		//	- Discrete linear programming.
		//		CBC, Cgl, SYMPHONY, ABACUS.
		//	- Continuous nonliner pLrogramming.
		//		Ipopt.
		//	- Discrete nonlinear programming.
		//		BONMIN.
		//	- Nonconvex nonlinear mixed integer programming.
		//		Couenne.
		//	- Semidefinite programming.
		//		CSDP.
		//	- Mathematical programming language.
		//		CMPL, GNU MathProg (GMPL), MPS format, LP format.
		//	- Automatic differentiation (AutoDiff).
		//		CppAD.
		//retval = coin_or_main(argc, argv);

		std::cout << "\nSolving Constraint Integer Programs (SCIP) Optimization Suite -------" << std::endl;
		//	- Linear programming.
		//		SoPlex.
		//	- Mixed integer programming.
		//		GCG, UG.
		//	- Mathematical programming language.
		//		ZIMPL, LP format, MPS format.
		//retval = scip_main(argc, argv);
	}
    catch (const std::bad_alloc &e)
	{
		std::cout << "std::bad_alloc caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cout << "Unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
