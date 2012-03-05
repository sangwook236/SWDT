//#include "stdafx.h"

#if !defined(M_PI)
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

// [ref] ${MOCAPY_ROOT}/examples/examples.cpp
void mocapy_discrete_hmm()
{
#if 1
	mocapy::mocapy_seed((uint)5556574);
#else
	mocapy::mocapy_seed((uint)std::time(NULL));
#endif

	// number of trainining sequences
	const int N = 100;

	// sequence lengths
	const int T = 100;

	// Gibbs sampling parameters
	const int MCMC_BURN_IN = 10;

	//---------------------------------------------------------------
	// HMM hidden and observed node sizes
	const uint H_SIZE = 2;
	const uint O_SIZE = 2;
	const bool init_random = false;

	mocapy::CPD th0_cpd;
	th0_cpd.set_shape(2); th0_cpd.set_values(mocapy::vec(0.1, 0.9));

	mocapy::CPD th1_cpd;
	th1_cpd.set_shape(2, 2); th1_cpd.set_values(mocapy::vec(0.95, 0.05, 0.1, 0.9));

	mocapy::CPD to0_cpd;
	to0_cpd.set_shape(2, 2); to0_cpd.set_values(mocapy::vec(0.1, 0.9, 0.8, 0.2));

	// the target DBN (this DBN generates the data)
	mocapy::Node *th0 = mocapy::NodeFactory::new_discrete_node(H_SIZE, "th0", init_random, th0_cpd);
	mocapy::Node *th1 = mocapy::NodeFactory::new_discrete_node(H_SIZE, "th1", init_random, th1_cpd);
	mocapy::Node *to0 = mocapy::NodeFactory::new_discrete_node(O_SIZE, "to0", init_random, to0_cpd);

	mocapy::DBN tdbn;
	tdbn.set_slices(mocapy::vec(th0, to0), mocapy::vec(th1, to0));

	tdbn.add_intra("th0", "to0");
	tdbn.add_inter("th0", "th1");
	tdbn.construct();

	//---------------------------------------------------------------
	// the model DBN (this DBN will be trained)
	// for mh0, get the CPD from th0 and fix parameters
	mocapy::Node *mh0 = mocapy::NodeFactory::new_discrete_node(H_SIZE, "mh0", init_random, mocapy::CPD(), th0, true);
	mocapy::Node *mh1 = mocapy::NodeFactory::new_discrete_node(H_SIZE, "mh1", init_random);
	mocapy::Node *mo0 = mocapy::NodeFactory::new_discrete_node(O_SIZE, "mo0", init_random);

	mocapy::DBN mdbn;
	mdbn.set_slices(mocapy::vec(mh0, mo0), mocapy::vec(mh1, mo0));

	mdbn.add_intra("mh0", "mo0");
	mdbn.add_inter("mh0", "mh1");
	mdbn.construct();

	std::cout << "*** TARGET ***" << std::endl;
	std::cout << *th0 << std::endl;
	std::cout << *th1 << std::endl;
	std::cout << *to0 << std::endl;

	std::cout << "*** MODEL ***" << std::endl;
	std::cout << *mh0 << std::endl;
	std::cout << *mh1 << std::endl;
	std::cout << *mo0 << std::endl;

	//---------------------------------------------------------------
	std::vector<mocapy::Sequence> seq_list;
	std::vector<mocapy::MDArray<mocapy::eMISMASK> > mismask_list;

	std::cout << "generating data" << std::endl;

	mocapy::MDArray<mocapy::eMISMASK> mismask;
	mismask.repeat(T, mocapy::vec(mocapy::MOCAPY_HIDDEN, mocapy::MOCAPY_OBSERVED));

	// generate the data
	double sum_LL(0);
	for (int i = 0; i < N; ++i)
	{
		std::pair<mocapy::Sequence, double> seq_ll = tdbn.sample_sequence(T);
		sum_LL += seq_ll.second;
		seq_list.push_back(seq_ll.first);
		mismask_list.push_back(mismask);
	}
	std::cout << "average LL: " << sum_LL/N << std::endl;

	//---------------------------------------------------------------
	mocapy::GibbsRandom mcmc = mocapy::GibbsRandom(&mdbn);
	mocapy::EMEngine em = mocapy::EMEngine(&mdbn, &mcmc, &seq_list, &mismask_list);

	mocapy::InfEngineMCMC inf = mocapy::InfEngineMCMC(&mdbn, &mcmc, &(seq_list[0]), mismask_list[0]);
	inf.initialize_viterbi_generator(5, 5, true);

	for (uint i = 0; i < 3; ++i)
	{
		mocapy::Sample s = inf.viterbi_next();
		std::cout << "Viterbi LL = " << s.ll << std::endl;
	}
	std::cout << "ending Viterbi" << std::endl;

	std::cout << "starting EM loop" << std::endl;
	double bestLL = -1000;
	uint it_no_improvement(0);
	uint i(0);
	// start EM loop
	while (it_no_improvement < 100)
	{
		em.do_E_step(1, MCMC_BURN_IN, true);

		const double ll = em.get_loglik();

		std::cout << "LL = " << ll;

		if (ll > bestLL)
		{
			std::cout << " * saving model *" << std::endl;
			mdbn.save("discrete_hmm.dbn");
			bestLL = ll;
			it_no_improvement = 0;
		}
		else
		{
			++it_no_improvement;
			std::cout << std::endl;
		}

		++i;
		em.do_M_step();
	}

	std::cout << "DONE" << std::endl;

	mdbn.load("discrete_hmm.dbn");

	std::cout << "*** TARGET ***" << std::endl;
	std::cout << *th0 << std::endl;
	std::cout << *th1 << std::endl;
	std::cout << *to0 << std::endl;

	std::cout << "*** MODEL ***" << std::endl;
	std::cout << *mh0 << std::endl;
	std::cout << *mh1 << std::endl;
	std::cout << *mo0 << std::endl;

	delete th0;
	delete th1;
	delete to0;

	delete mh0;
	delete mh1;
	delete mo0;
}
