//#include "stdafx.h"
#include <pnl_dll.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>


namespace {
namespace local {

pnl::CDBN * create_simple_hmm()
{
/*
	a simple HMM

		X0 -> X1
		|     |
		v     v
		Y0    Y1

	where
		X0, X1 - bivariate tabular nodes
		Y0, Y1 - trivariate tabular nodes
*/

/*
	states = ('Rainy', 'Sunny')

	observations = ('walk', 'shop', 'clean')

	start_probability = {'Rainy': 0.6, 'Sunny': 0.4}

	transition_probability = {
	   'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
	   'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
	}

	emission_probability = {
	   'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
	   'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
	}
*/

	// create static model
	const int numNodes = 4;  // number of nodes
	const int numNodeTypes = 2;  // number of node types (all nodes are discrete)

	// specify the graph structure of the model
	const int numNeighs[] = { 2, 1, 2, 1 };

	const int neigh0[] = { 1, 2 };
	const int neigh1[] = { 0 };
	const int neigh2[] = { 0, 3 };
	const int neigh3[] = { 2 };
    const int *neighs[] = { neigh0, neigh1, neigh2, neigh3 };

    const pnl::ENeighborType orient0[] = { pnl::ntChild, pnl::ntChild };
    const pnl::ENeighborType orient1[] = { pnl::ntParent };
    const pnl::ENeighborType orient2[] = { pnl::ntParent, pnl::ntChild };
    const pnl::ENeighborType orient3[] = { pnl::ntParent };
    const pnl::ENeighborType *orients[] = { orient0, orient1, orient2, orient3 };

	pnl::CGraph *graph = pnl::CGraph::Create(numNodes, numNeighs, neighs, orients);

	// create static BNet
	pnl::nodeTypeVector nodeTypes(numNodeTypes);
	nodeTypes[0].SetType(true, 2);  // discrete & binary
	nodeTypes[1].SetType(true, 3);  // continuous & trinary

	pnl::intVector nodeAssociation(numNodes);
	nodeAssociation[0] = 0;
	nodeAssociation[1] = 1;
	nodeAssociation[2] = 0;
	nodeAssociation[3] = 1;

	pnl::CBNet *hmm_bnet = pnl::CBNet::Create(numNodes, nodeTypes, nodeAssociation, graph);

	// create raw data tables for CPDs
	const float table0[] = { 0.6f, 0.4f };
	const float table1[] = { 0.1f, 0.4f, 0.5f, 0.6f, 0.3f, 0.1f };
	const float table2[] = { 0.7f, 0.3f, 0.4f, 0.6f };
	const float table3[] = { 0.1f, 0.4f, 0.5f, 0.6f, 0.3f, 0.1f };

	// create factors and attach their to model
	hmm_bnet->AllocFactors();
	hmm_bnet->AllocFactor(0);
	hmm_bnet->GetFactor(0)->AllocMatrix(table0, pnl::matTable);
	hmm_bnet->AllocFactor(1);
	hmm_bnet->GetFactor(1)->AllocMatrix(table1, pnl::matTable);
	hmm_bnet->AllocFactor(2);
	hmm_bnet->GetFactor(2)->AllocMatrix(table2, pnl::matTable);
	hmm_bnet->AllocFactor(3);
	hmm_bnet->GetFactor(3)->AllocMatrix(table3, pnl::matTable);

	// create DBN
	return pnl::CDBN::Create(hmm_bnet);
}

// [ref]
//	CompareMPE() in ${PNL_ROOT}/c_pgmtk/tests/src/AJTreeInfDBN.cpp
//	CompareViterbyArHMM() in ${PNL_ROOT}/c_pgmtk/tests/src/A1_5JTreeInfDBNCondGauss.cpp
void infer_mpe_in_hmm(const boost::scoped_ptr<pnl::CDBN> &hmm)
{
	//
	const pnl::intVector obsNodes(1, 1);  // 1st node ==> observed node
	pnl::valueVector obsNodesVals(1);
#if 0
	const int observations[] = { 0, 1, 2, 0, 1, 2, 2, 2 };
#elif 1
	const int observations[] = { 2, 1, 0, 0, 2, 1 };
#else
	const int observations[] = { 0, 1, 2 };
#endif

	// number of time slices for unrolling
	const int numTimeSlices = sizeof(observations) / sizeof(observations[0]);

	// create evidence for every time-slice
	pnl::pEvidencesVector evidences(numTimeSlices);
	for (int time_slice = 0; time_slice < numTimeSlices; ++time_slice)
	{
		obsNodesVals[0].SetInt(observations[time_slice]);
		evidences[time_slice] = pnl::CEvidence::Create(hmm.get(), obsNodes, obsNodesVals);
	}

	// create an inference engine
	boost::scoped_ptr<pnl::C1_5SliceJtreeInfEngine> inferEng(pnl::C1_5SliceJtreeInfEngine::Create(hmm.get()));

	// create inference (smoothing) for DBN
	inferEng->DefineProcedure(pnl::ptViterbi, numTimeSlices);
	inferEng->EnterEvidence(&evidences.front(), numTimeSlices);
	inferEng->FindMPE();

	pnl::intVector queryPrior(1), query(2);
	queryPrior[0] = 0;  // 0th node ==> hidden state
	query[0] = 0;  query[1] = 2;  // 0th & 2nd nodes ==> hidden states

	for (int time_slice = 0; time_slice < numTimeSlices; ++time_slice)
	{
		if (time_slice)  // for the transition network
		{
			inferEng->MarginalNodes(&query.front(), query.size(), time_slice);
		}
		else  // for the prior network
		{
			inferEng->MarginalNodes(&queryPrior.front(), queryPrior.size(), time_slice);
		}

		const pnl::CPotential *queryMPE = inferEng->GetQueryMPE();

		std::cout << ">>> query time-slice: " << time_slice << std::endl;

		int numNodes = 0;
		const int *domain = NULL;
		queryMPE->GetDomain(&numNodes, &domain);

		std::cout << " domain: ";
		for (int i = 0; i < numNodes; ++i)
		{
			std::cout << domain[i] << " ";
		}
		std::cout << std::endl;

		// TODO [check] >> is this code really correct?
		const pnl::CEvidence *mpe = inferEng->GetMPE();
		std::cout << " MPE node value: ";
#if 0
		for (int i = 0; i < numNodes; ++i)
		{
			const int mpeNodeVal = mpe->GetValue(domain[i])->GetInt();
			std::cout << mpeNodeVal << " ";
		}
		std::cout << std::endl;
#else
		const int mpeNodeVal = mpe->GetValue(domain[numNodes-1])->GetInt();
		std::cout << mpeNodeVal << std::endl;
#endif
	}

	//
	for (int i = 0; i < numTimeSlices; ++i)
	{
		delete evidences[i];
	}
}

}  // namespace local
}  // unnamed namespace

void viterbi_segmentation()
{
	std::cout << "========== simple HMM" << std::endl;
	{
		const boost::scoped_ptr<pnl::CDBN> hmm(local::create_simple_hmm());

		if (!hmm)
		{
			std::cout << "fail to create a probabilistic graphical model at " << __LINE__ << " in " << __FILE__ << std::endl;
			return;
		}

		local::infer_mpe_in_hmm(hmm);
	}
}
