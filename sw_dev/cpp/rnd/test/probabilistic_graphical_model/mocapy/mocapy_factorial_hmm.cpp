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

// [ref] ${MOCAPY_HOME}/examples/hmm_factorial.cpp
void factorial_hmm()
{
#if 1
	mocapy::mocapy_seed((uint)5556574);
#else
	mocapy::mocapy_seed((uint)std::time(NULL));
#endif

	// HMM hidden and observed node sizes
	const uint H_SIZE = 5;
	const uint O_SIZE = 3;

	mocapy::Node *th1 = mocapy::NodeFactory::new_discrete_node(H_SIZE, "th1");
	mocapy::Node *th2 = mocapy::NodeFactory::new_discrete_node(H_SIZE, "th2");
	mocapy::Node *th3 = mocapy::NodeFactory::new_discrete_node(H_SIZE, "th3");

	mocapy::Node *th01 = mocapy::NodeFactory::new_discrete_node(H_SIZE, "th01");
	mocapy::Node *th02 = mocapy::NodeFactory::new_discrete_node(H_SIZE, "th02");
	mocapy::Node *th03 = mocapy::NodeFactory::new_discrete_node(H_SIZE, "th03");

	mocapy::Node *to = mocapy::NodeFactory::new_discrete_node(O_SIZE, "to");

	mocapy::DBN tdbn;
	tdbn.set_slices(mocapy::vec(th01, th02, th03, to), mocapy::vec(th1, th2, th3, to));

	tdbn.add_intra("th1", "to");
	tdbn.add_intra("th2", "to");
	tdbn.add_intra("th3", "to");
	tdbn.add_inter("th1", "th01");
	tdbn.add_inter("th2", "th02");
	tdbn.add_inter("th3", "th03");
	tdbn.construct();
}

}  // namespace my_mocapy
