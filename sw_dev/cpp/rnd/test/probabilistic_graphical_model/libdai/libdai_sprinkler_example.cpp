//#include "stdafx.h"
#include <dai/factorgraph.h>
#include <iostream>
#include <fstream>
#include <vector>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_libdai {

// REF [file] >> ${LIBDAI_HOME}/examples/example_sprinkler.cpp
void sprinkler_example()
{
	// This example program illustrates how to construct a factorgraph by means of the sprinkler network example discussed at
	// http://www.cs.ubc.ca/~murphyk/Bayes/bnintro.html

	dai::Var C(0, 2);  // Define binary variable Cloudy (with label 0).
	dai::Var S(1, 2);  // Define binary variable Sprinkler (with label 1).
	dai::Var R(2, 2);  // Define binary variable Rain (with label 2).
	dai::Var W(3, 2);  // Define binary variable Wetgrass (with label 3).

	// Define probability distribution for C.
	dai::Factor P_C(C);
	P_C.set(0, 0.5);  // C = 0
	P_C.set(1, 0.5);  // C = 1

	// Define conditional probability of S given C.
	dai::Factor P_S_given_C(dai::VarSet(S, C));
	P_S_given_C.set(0, 0.5);  // C = 0, S = 0
	P_S_given_C.set(1, 0.9);  // C = 1, S = 0
	P_S_given_C.set(2, 0.5);  // C = 0, S = 1
	P_S_given_C.set(3, 0.1);  // C = 1, S = 1

	// Define conditional probability of R given C.
	dai::Factor P_R_given_C(dai::VarSet(R, C));
	P_R_given_C.set(0, 0.8);  // C = 0, R = 0
	P_R_given_C.set(1, 0.2);  // C = 1, R = 0
	P_R_given_C.set(2, 0.2);  // C = 0, R = 1
	P_R_given_C.set(3, 0.8);  // C = 1, R = 1

	// Define conditional probability of W given S and R.
	dai::Factor P_W_given_S_R(dai::VarSet(S, R) | W);
	P_W_given_S_R.set(0, 1.0);  // S = 0, R = 0, W = 0
	P_W_given_S_R.set(1, 0.1);  // S = 1, R = 0, W = 0
	P_W_given_S_R.set(2, 0.1);  // S = 0, R = 1, W = 0
	P_W_given_S_R.set(3, 0.01);  // S = 1, R = 1, W = 0
	P_W_given_S_R.set(4, 0.0);  // S = 0, R = 0, W = 1
	P_W_given_S_R.set(5, 0.9);  // S = 1, R = 0, W = 1
	P_W_given_S_R.set(6, 0.9);  // S = 0, R = 1, W = 1
	P_W_given_S_R.set(7, 0.99);  // S = 1, R = 1, W = 1

	// Build factor graph consisting of those four factors.
	std::vector<dai::Factor> SprinklerFactors;
	SprinklerFactors.push_back(P_C);
	SprinklerFactors.push_back(P_R_given_C);
	SprinklerFactors.push_back(P_S_given_C);
	SprinklerFactors.push_back(P_W_given_S_R);
	dai::FactorGraph SprinklerNetwork(SprinklerFactors);

	// Write factorgraph to a file.
	SprinklerNetwork.WriteToFile("./data/probabilistic_graphical_model/libdai/sprinkler.fg");
	std::cout << "Sprinkler network written to sprinkler.fg." << std::endl;

	// Output some information about the factorgraph.
	std::cout << SprinklerNetwork.nrVars() << " variables" << std::endl;
	std::cout << SprinklerNetwork.nrFactors() << " factors" << std::endl;

	// Calculate joint probability of all four variables.
	dai::Factor P;
	for (std::size_t I = 0; I < SprinklerNetwork.nrFactors(); ++I)
		P *= SprinklerNetwork.factor(I);
	//P.normalize();  // Not necessary: a Bayesian network is already normalized by definition.

	// Calculate some probabilities.
	dai::Real denom = P.marginal(W)[1];
	std::cout << "P(W=1) = " << denom << std::endl;
	std::cout << "P(S=1 | W=1) = " << P.marginal(dai::VarSet(S, W))[3] / denom << std::endl;
	std::cout << "P(R=1 | W=1) = " << P.marginal(dai::VarSet(R, W))[3] / denom << std::endl;
}

}  // namespace my_libdai
