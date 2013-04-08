//#include "stdafx.h"

#if defined(_MSC_VER) && !defined(M_PI)
#include <boost/math/constants/constants.hpp>
#define M_PI (boost::math::constants::pi<double>())
#endif

#include <mocapy.h>
#include <iostream>
#include <ctime>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_mocapy {

// [ref] ${MOCAPY_HOME}/examples/hmm_simple.cpp
void simple_hmm()
{
	// the dynamic Bayesian network
	mocapy::DBN dbn;

	// nodes in slice 1
	mocapy::Node *h1 = mocapy::NodeFactory::new_discrete_node(5, "h1");
	mocapy::Node *o1 = mocapy::NodeFactory::new_discrete_node(2, "o1");

	// nodes in slice 2
	mocapy::Node *h2 = mocapy::NodeFactory::new_discrete_node(5, "h2");

	// set architecture
	dbn.set_slices(mocapy::vec(h1, o1), mocapy::vec(h2, o1));

	dbn.add_intra("h1", "o1");
	dbn.add_inter("h1", "h2");
	dbn.construct();

	//---------------------------------------------------------------
	std::cout << "loading traindata" << std::endl;
	mocapy::GibbsRandom mcmc = mocapy::GibbsRandom(&dbn);

	mocapy::EMEngine em = mocapy::EMEngine(&dbn, &mcmc);
	em.load_mismask("./probabilistic_graphical_model_data/mocapy/mismask.dat");  // [ref] Mocapy++ manual, pp. 18~19.
	em.load_weights("./probabilistic_graphical_model_data/mocapy/weights.dat");  // [ref] Mocapy++ manual, pp. 23.
	em.load_sequences("./probabilistic_graphical_model_data/mocapy/traindata.dat");  // [ref] Mocapy++ manual, pp. 19.

	std::cout << "starting EM loop" << std::endl;
	for (uint i = 0; i < 100; ++i)
	{
		em.do_E_step(20, 10);

		const double ll = em.get_loglik();
		std::cout << "LL = " << ll << std::endl;

		em.do_M_step();
	}

	std::cout << "h1: " << *h1 << std::endl;
	std::cout << "o1: " << *o1 << std::endl;
	std::cout << "h2: " << *h2 << std::endl;
}

}  // namespace my_mocapy
