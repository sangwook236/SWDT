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

// [ref] ${MOCAPY_HOME}/examples/infenginemm_example.cpp
void mixture_model_inference()
{
    // Mocapy++ provides exact methods for doing inference in mixture models (MMs).
    //  [ref] Mocapy++ manual, pp. 27~28.

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

	// set cpd for hidde node
	const double hd_cpd_array[] =
	{
		1.0, 0.0, 0.0, 0.0,
		0.25, 0.25, 0.25, 0.25
	};
	mocapy::CPD hd_cpd;
	hd_cpd.set_shape(2, 4);
	local::copy_flat(hd_cpd.flat_iterator(), hd_cpd_array);

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

	mocapy::DiscreteNode *hd = new mocapy::DiscreteNode();
	hd->set_densities(mocapy::DiscreteDensities(4, hd_cpd));

	mocapy::DiscreteNode *out = new mocapy::DiscreteNode();
	out->set_densities(mocapy::DiscreteDensities(4, out_cpd));

	//---------------------------------------------------------------
	// setup dbn
	std::vector<mocapy::Node *> nodes = local::vect(in->base(), hd->base(), out->base());

	mocapy::DBN dbn(nodes, nodes);

	dbn.add_intra(0, 1);
	dbn.add_intra(1, 2);
	dbn.construct();

	// output the CPDs
	std::cout << "in:  \n" << in->get_densities()->getCPD() << std::endl;
	std::cout << "hd:  \n" << hd->get_densities()->getCPD() << std::endl;
	std::cout << "out: \n" << out->get_densities()->getCPD() << std::endl;
	std::cout << std::endl;

	//---------------------------------------------------------------
	std::cout << "******************" << std::endl;
	std::cout << "Setting up dataset" << std::endl;
	std::cout << "******************" << std::endl;

	mocapy::Sequence data;
	data.set_shape(5, 3);
	const double seq_array[] = {
		1, 0, 1,
		1, 0, 1,
		1, 0, 1,
		1, 0, 1,
		1, 0, 1
	};
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

	std::cout << "Data: \n" << data << std::endl;
	std::cout << "Mism: \n" << mism << std::endl;
	std::cout << "Mism_sample: \n" << mism_sample << std::endl;

	//---------------------------------------------------------------
	std::cout << "*************************" << std::endl;
	std::cout << "Example of SampleInfEngineMM" << std::endl;
	std::cout << "*************************" << std::endl;

	// setup the sampler
	mocapy::SampleInfEngineMM sampler(&dbn, data, mism_sample, 1);

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

	std::cout << "undo()" << std::endl;
	sampler.undo();

	// use the other mismask, so that we can just sample the hidden states
	sampler.set_seq_mismask(data, mism);
	sampler.set_start_end();

	sample = sampler.sample_next();
	std::cout << "Sample only hidden states:\n" << sample;
	std::cout << "ll of sample = " << sampler.calc_ll(mism) << std::endl << std::endl;

	//---------------------------------------------------------------
	std::cout << "********************************" << std::endl;
	std::cout << "Example of LikelihoodInfEngineMM" << std::endl;
	std::cout << "********************************" << std::endl;

	mocapy::LikelihoodInfEngineMM infengine(&dbn, 1);

	const double ll1 = infengine.calc_ll(data, mism, false);
	std::cout << "ll of initail values = " << ll1 << std::endl;

	const double ll2 = infengine.calc_ll(data, mism, 0, -1, true);
	std::cout << "ll of initail values (including parents) = " << ll2 << std::endl;

	//---------------------------------------------------------------
	// clean up
	delete in;
	delete hd;
	delete out;
}

}  // namespace my_mocapy
