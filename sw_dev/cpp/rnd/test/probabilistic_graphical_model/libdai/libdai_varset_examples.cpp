//#include "stdafx.h"
#include <dai/varset.h>
#include <dai/index.h>
#include <iostream>
#include <fstream>
#include <map>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_libdai {

// REF [file] >> ${LIBDAI_HOME}/examples/example_varset.cpp
void varset_example()
{
	dai::Var x0(0, 2);  // Define binary variable x0 (with label 0)
	dai::Var x1(1, 3);  // Define ternary variable x1 (with label 1)

	// Define set X = {x0, x1}
	dai::VarSet X;  // empty
	X |= x1;  // X = {x1}
	X |= x0;  // X = {x1, x0}
	std::cout << "X = " << X << std::endl << std::endl;  // Note that the elements of X are ordered according to their labels

	// Output some information about x0, x1 and X
	std::cout << "Var " << x0 << " has " << x0.states() << " states (possible values)." << std::endl;
	std::cout << "Var " << x1 << " has " << x1.states() << " states." << std::endl << std::endl;
	std::cout << "VarSet " << X << " has " << X.nrStates() << " states (joint assignments of its variables)." << std::endl << std::endl;

	std::cout << "States of VarSets correspond to states of their constituent Vars:" << std::endl;
	std::cout << "  state of x0:   state of x1:   (linear) state of X:" << std::endl;
	for (std::size_t s1 = 0; s1 < x1.states(); ++s1)  // for all states s1 of x1
		for (std::size_t s0 = 0; s0 < x0.states(); ++s0)  // for all states s0 of x0
		{
			// store s0 and s1 in a map "states"
			std::map<dai::Var, std::size_t> states;
			states[x0] = s0;
			states[x1] = s1;

			// output states of x0, x1 and corresponding state of X
			std::cout << "    " << s0 << "              " << s1 << "              " << dai::calcLinearState(X, states) << std::endl;

			// calcState() is the inverse of calcLinearState()
			DAI_ASSERT(dai::calcState(X, dai::calcLinearState(X, states)) == states);
		}

	std::cout << std::endl << "And vice versa:" << std::endl;
	std::cout << "  state of x0:   state of x1:   (linear) state of X:" << std::endl;
	//--S [] 2017/05/16 : Sang-Wook Lee.
	//for (std::size_t S = 0; S < X.nrStates(); ++S)  // for all (joint) states of X
	for (std::size_t S = 0; S < X.nrStates().get_ui(); ++S)  // for all (joint) states of X
	//--E [] 2017/05/16 : Sang-Wook Lee.
	{
		// calculate states of x0 and x1 corresponding to state S of X
		std::map<dai::Var, std::size_t> states = dai::calcState(X, S);

		// output state of X and corresponding states of x0, x1
		std::cout << "    " << states[x0] << "              " << states[x1] << "              " << S << std::endl;

		// calcLinearState() is the inverse of calcState()
		DAI_ASSERT(dai::calcLinearState(X, dai::calcState(X, S)) == S);
	}

	std::cout << std::endl << "Iterating over all joint states using the State class:" << std::endl;
	std::cout << "  state of x0:   state of x1:   (linear) state of X:   state of X (as a map):" << std::endl;
	for (dai::State S(X); S.valid(); ++S)
	{
		// output state of X and corresponding states of x0, x1
		std::cout << "    " << S(x0) << "              " << S(x1) << "              " << S << "                      " << S.get() << std::endl;
	}
}

}  // namespace my_libdai
