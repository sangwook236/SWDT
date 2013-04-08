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

template<typename T>
std::vector<T> vect(T t1, T t2, T t3)
{
	std::vector<T> v;
	v.push_back(t1);
	v.push_back(t2);
	v.push_back(t3);
	return v;
}

void copy_flat(const std::vector<double *> &flat_iterator, const double *array)
{
	for (uint i = 0; i < flat_iterator.size(); ++i)
	{
		*flat_iterator[i] = array[i];
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_mocapy {

// [ref] ${MOCAPY_HOME}/examples/infenginehmm_example.cpp
void hmm_inference()
{
	mocapy::mocapy_seed((uint)std::time(NULL));

	//---------------------------------------------------------------
	std::cout << "******************" << std::endl;
	std::cout << "Setting up network" << std::endl;
	std::cout << "******************" << std::endl;

	// set cpd for in-node
	const double in_cpd_array[] = { 0.5, 0.5 };
	mocapy::CPD in_cpd;
	in_cpd.set_shape(2);
	local::copy_flat(in_cpd.flat_iterator(), in_cpd_array);

	// set cpd for hd0
	const double hd0_cpd_array[] = {
		1.0, 0.0, 0.0, 0.0,
		0.25, 0.25, 0.25, 0.25
	};
	mocapy::CPD hd0_cpd;
	hd0_cpd.set_shape(2, 4);
	local::copy_flat(hd0_cpd.flat_iterator(), hd0_cpd_array);

	// set cpd for hd1
	const double hd1_cpd_array[] = {
		0, 1, 0, 0,
		0.25, 0.25, 0.25, 0.25,
		0, 0, 1, 0,
		0.25, 0.25, 0.25, 0.25,
		0, 0, 0, 1,
		0.25, 0.25, 0.25, 0.25,
		1, 0, 0, 0,
		0.25, 0.25, 0.25, 0.25
	};

	mocapy::CPD hd1_cpd;
	hd1_cpd.set_shape(4, 2, 4);
	local::copy_flat(hd1_cpd.flat_iterator(), hd1_cpd_array);

	// set cpd for out node
	const double out_cpd_array[] = {
		0.7, 0.0, 0.0, 0.0,
		0.0, 0.7, 0.0, 0.0,
		0.0, 0.0, 0.7, 0.0,
		0.0, 0.0, 0.0, 0.7
	};

	mocapy::CPD out_cpd;
	out_cpd.set_shape(4, 4);
	local::copy_flat(out_cpd.flat_iterator(), out_cpd_array);

	// setup nodes
	mocapy::DiscreteNode *in = new mocapy::DiscreteNode();
	in->set_densities(mocapy::DiscreteDensities(2, in_cpd));

	mocapy::DiscreteNode *hd0 = new mocapy::DiscreteNode();
	hd0->set_densities(mocapy::DiscreteDensities(4, hd0_cpd));

	mocapy::DiscreteNode *hd1 = new mocapy::DiscreteNode();
	hd1->set_densities(mocapy::DiscreteDensities(4, hd1_cpd));

	mocapy::DiscreteNode *out = new mocapy::DiscreteNode();
	out->set_densities(mocapy::DiscreteDensities(4, out_cpd));

	//---------------------------------------------------------------
	// setup dbn
	std::vector<mocapy::Node *> start_nodes = local::vect(in->base(), hd0->base(), out->base());
	std::vector<mocapy::Node *> end_nodes = local::vect(in->base(), hd1->base(), out->base());

	mocapy::DBN dbn(start_nodes, end_nodes);

	dbn.add_intra(0, 1);
	dbn.add_intra(1, 2);
	dbn.add_inter(1, 1);
	dbn.construct();

	// output the CPDs
	std::cout << "in:  \n" << in->get_densities()->getCPD() << std::endl;
	std::cout << "hd0: \n" << hd0->get_densities()->getCPD() << std::endl;
	std::cout << "hd1: \n" << hd1->get_densities()->getCPD() << std::endl;
	std::cout << "out:  \n" << out->get_densities()->getCPD() << std::endl;
	std::cout << std::endl;

	//---------------------------------------------------------------
	std::cout << "******************" << std::endl;
	std::cout << "Setting up dataset" << std::endl;
	std::cout << "******************" << std::endl;

	mocapy::Sequence data;
	data.set_shape(5, 3);
	const double seq_array[] = {
		1, 0, 1,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1};
	local::copy_flat(data.flat_iterator(), seq_array);

	mocapy::MDArray<mocapy::eMISMASK> mism;
	mism.set_shape(5, 3);
	mism.set_wildcard(mocapy::vec(-1, 0), mocapy::MOCAPY_OBSERVED);
	mism.set_wildcard(mocapy::vec(-1, 1), mocapy::MOCAPY_HIDDEN);
	mism.set_wildcard(mocapy::vec(-1, 2), mocapy::MOCAPY_OBSERVED);

	mocapy::MDArray<mocapy::eMISMASK> mism_sample;
	mism_sample.set_shape(5, 3);
	mism_sample.set_wildcard(mocapy::vec(-1, 0), mocapy::MOCAPY_OBSERVED);
	mism_sample.set_wildcard(mocapy::vec(-1, 1), mocapy::MOCAPY_HIDDEN);
	mism_sample.set_wildcard(mocapy::vec(-1, 2), mocapy::MOCAPY_HIDDEN);

	mism_sample.get(3, 2) = mocapy::MOCAPY_OBSERVED;

	std::cout << "Data: \n" << data << std::endl;
	std::cout << "Mism: \n" << mism << std::endl;
	std::cout << "Mism_sample: \n" << mism_sample << std::endl;

	//---------------------------------------------------------------
	std::cout << "*****************************" << std::endl;
	std::cout << "Example of SampleInfEngineHMM" << std::endl;
	std::cout << "*****************************" << std::endl;

	dbn.randomGen->get_rand();
	std::cout << "TEST" << std::endl;

	// setup the sampler
	mocapy::SampleInfEngineHMM sampler(&dbn, data, mism_sample, 1);

	mocapy::MDArray<double> sample = sampler.sample_next();
	std::cout << "Sample:\n" << sample;
	std::cout << "ll of sample = " << sampler.calc_ll(mism) << std::endl << std::endl;

	std::cout << "undo()" << std::endl;
	sampler.undo();
	std::cout << "ll of initial values =" << sampler.calc_ll(mism) << std::endl << std::endl;

	std::cout << "Setting start=0 and end=1" << std::endl << std::endl;
	sampler.set_start_end(0, 1);

	sample = sampler.sample_next();
	std::cout << "Sample:\n" << sample;
	std::cout << "ll of sample = " << sampler.calc_ll(mism) << std::endl << std::endl;

	//---------------------------------------------------------------
	std::cout << "*********************************" << std::endl;
	std::cout << "Example of LikelihoodInfEngineHMM" << std::endl;
	std::cout << "*********************************" << std::endl;

	mocapy::LikelihoodInfEngineHMM infengine(&dbn, 1);

	const double ll = infengine.calc_ll(data, mism);
	std::cout << "ll of initail values = " << ll << std::endl;

	//---------------------------------------------------------------
	// clean up
	delete in;
	delete hd0;
	delete hd1;
	delete out;
}

}  // namespace my_mocapy
