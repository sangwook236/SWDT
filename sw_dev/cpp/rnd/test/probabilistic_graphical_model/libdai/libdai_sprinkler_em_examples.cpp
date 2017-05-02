//#include "stdafx.h"
#include <dai/alldai.h>
#include <fstream>
#include <iostream>
#include <string>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_libdai {

// REF [file] >> ${LIBDAI_HOME}/examples/example_sprinkler_em.cpp
void sprinkler_em_example()
{
	// This example program illustrates how to learn the parameters of a Bayesian network from a sample of the sprinkler network discussed at
	// http://www.cs.ubc.ca/~murphyk/Bayes/bnintro.html

	// The factor graph file (sprinkler.fg) has to be generated first by running example_sprinkler, and the data sample file (sprinkler.tab) by running example_sprinkler_gibbs

	// Read the factorgraph from the file
	dai::FactorGraph SprinklerNetwork;
	SprinklerNetwork.ReadFromFile("./data/probabilistic_graphical_model/libdai/sprinkler.fg");

	// Prepare junction-tree object for doing exact inference for E-step
	dai::PropertySet infprops;
	infprops.set("verbose", (std::size_t)1);
	infprops.set("updates", std::string("HUGIN"));
	dai::InfAlg *inf = dai::newInfAlg("JTREE", SprinklerNetwork, infprops);
	inf->init();

	// Read sample from file
	dai::Evidence e;
	std::ifstream estream("./data/probabilistic_graphical_model/libdai/sprinkler.tab");
	e.addEvidenceTabFile(estream, SprinklerNetwork);
	std::cout << "Number of samples: " << e.nrSamples() << std::endl;

	// Read EM specification
	std::ifstream emstream("./data/probabilistic_graphical_model/libdai/sprinkler.em");
	dai::EMAlg em(e, *inf, emstream);

	// Iterate EM until convergence
	while (!em.hasSatisfiedTermConditions())
	{
		dai::Real l = em.iterate();
		std::cout << "Iteration " << em.Iterations() << " likelihood: " << l <<std::endl;
	}

	// Output true factor graph
	std::cout << std::endl << "True factor graph:" << std::endl << "##################" << std::endl;
	std::cout.precision(12);
	std::cout << SprinklerNetwork;

	// Output learned factor graph
	std::cout << std::endl << "Learned factor graph:" << std::endl << "#####################" << std::endl;
	std::cout.precision(12);
	std::cout << inf->fg();

	// Clean up
	delete inf;
}

}  // namespace my_libdai
