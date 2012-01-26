#include "stdafx.h"
#include <pnl_dll.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>


namespace {
namespace local {

pnl::CDBN * create_simple_hmm()
{
#if 1
/*
	a simple HMM
		X0 -> X1
		|     | 
		v     v
		Y0    Y1 
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
	const int numNeigh[] = { 2, 1, 2, 1 };
	const int neigh0[] = { 1, 2 };
	const int neigh1[] = { 0 };
	const int neigh2[] = { 0, 3 };
	const int neigh3[] = { 2 };
    const int *neigh[] = { neigh0, neigh1, neigh2, neigh3 };

    const pnl::ENeighborType orient0[] = { pnl::ntChild, pnl::ntChild };
    const pnl::ENeighborType orient1[] = { pnl::ntParent };
    const pnl::ENeighborType orient2[] = { pnl::ntParent, pnl::ntChild };
    const pnl::ENeighborType orient3[] = { pnl::ntParent };
    const pnl::ENeighborType *orient[] = { orient0, orient1, orient2, orient3 };
	
	pnl::CGraph *pGraph = pnl::CGraph::Create(numNodes, numNeigh, neigh, orient);

	// create static BNet
	pnl::nodeTypeVector nodeTypes(numNodeTypes);
	nodeTypes[0].SetType(true, 2);
	nodeTypes[1].SetType(true, 3);

	pnl::intVector nodeAssociation(numNodes);
	nodeAssociation[0] = 0;
	nodeAssociation[1] = 1;
	nodeAssociation[2] = 0;
	nodeAssociation[3] = 1;
	
	pnl::CBNet *pBNet = pnl::CBNet::Create(numNodes, nodeTypes, nodeAssociation, pGraph);

	// create raw data tables for CPDs
	const float table0[] = { 0.6f, 0.4f };
	const float table1[] = { 0.1f, 0.4f, 0.5f, 0.6f, 0.3f, 0.1f };
	const float table2[] = { 0.7f, 0.3f, 0.4f, 0.6f };
	const float table3[] = { 0.1f, 0.4f, 0.5f, 0.6f, 0.3f, 0.1f };

	// create factors and attach their to model
	pBNet->AllocFactors();
	pBNet->AllocFactor(0); 
	pBNet->GetFactor(0)->AllocMatrix(table0, pnl::matTable);
	pBNet->AllocFactor(1);
	pBNet->GetFactor(1)->AllocMatrix(table1, pnl::matTable);
	pBNet->AllocFactor(2);
	pBNet->GetFactor(2)->AllocMatrix(table2, pnl::matTable);
	pBNet->AllocFactor(3);
	pBNet->GetFactor(3)->AllocMatrix(table3, pnl::matTable);
#else
	pnl::CBNet *pBNet = pnl::pnlExCreateRndArHMM();
#endif;

	// create DBN
	pnl::CDBN *pDBN = pnl::CDBN::Create(pBNet);

	return pDBN;
}

pnl::CDBN * create_hmm_with_ar_gaussian_observations()
{
/*
	an HMM with autoregressive Gaussian observations
		X0 -> X1
		|     | 
		v     v
		Y0 -> Y1 
*/

	// create static model
	const int numNodes = 4;  // number of nodes    
	const int numNodeTypes = 2;  // number of node types (all nodes are discrete)

	pnl::CNodeType *nodeTypes = new pnl::CNodeType [numNodeTypes];
	nodeTypes[0].SetType(true, 2);
	nodeTypes[1].SetType(false, 1);

	const int nodeAssociation[] = { 0, 1, 0, 1 };

	// create a DAG
	const int numNeigh[] = { 2, 2, 2, 2 };
	const int neigh0[] = { 1, 2 };
	const int neigh1[] = { 0, 3 };
	const int neigh2[] = { 0, 3 };
	const int neigh3[] = { 1, 2 };
	const int *neigh[] = { neigh0, neigh1, neigh2, neigh3 };

	const pnl::ENeighborType orient0[] = { pnl::ntChild, pnl::ntChild };
	const pnl::ENeighborType orient1[] = { pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient2[] = { pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient3[] = { pnl::ntParent, pnl::ntParent };
	const pnl::ENeighborType *orient[] = { orient0, orient1, orient2, orient3 };

	pnl::CGraph *pGraph = pnl::CGraph::Create(numNodes, numNeigh, neigh, orient);

	// create static BNet
	pnl::CBNet *pBNet = pnl::CBNet::Create(numNodes, numNodeTypes, nodeTypes, nodeAssociation, pGraph);
	pBNet->AllocFactors();

	// let arbitrary distribution is
	const float tableNode0[] = { 0.95f, 0.05f };
	const float tableNode2[] = { 0.1f, 0.9f, 0.5f, 0.5f };

	const float mean1w0 = -3.2f;  const float cov1w0 = 0.00002f; 
	const float mean1w1 = -0.5f;  const float cov1w1 = 0.0001f;

	const float mean3w0 = 6.5f;  const float cov3w0 = 0.03f;  const float weight3w0 = 1.0f;
	const float mean3w1 = 7.5f;  const float cov3w1 = 0.04f;  const float weight3w1 = 0.5f;

	pBNet->AllocFactor(0);
	pBNet->GetFactor(0)->AllocMatrix(tableNode0, pnl::matTable);

	pBNet->AllocFactor(1);
	int parent[] = { 0 };
	pBNet->GetFactor(1)->AllocMatrix(&mean1w0, pnl::matMean, -1, parent);
	pBNet->GetFactor(1)->AllocMatrix(&cov1w0, pnl::matCovariance, -1, parent);
	parent[0] = 1;
	pBNet->GetFactor(1)->AllocMatrix(&mean1w1, pnl::matMean, -1, parent);
	pBNet->GetFactor(1)->AllocMatrix(&cov1w1, pnl::matCovariance, -1, parent);

	pBNet->AllocFactor(2);
	pBNet->GetFactor(2)->AllocMatrix(tableNode2, pnl::matTable);

	pBNet->AllocFactor(3);
	parent[0] = 0;
	pBNet->GetFactor(3)->AllocMatrix(&mean3w0, pnl::matMean, -1, parent);
	pBNet->GetFactor(3)->AllocMatrix(&cov3w0, pnl::matCovariance, -1, parent);
	pBNet->GetFactor(3)->AllocMatrix(&weight3w0, pnl::matWeights, 0, parent);
	parent[0] = 1;
	pBNet->GetFactor(3)->AllocMatrix(&mean3w1, pnl::matMean, -1, parent);
	pBNet->GetFactor(3)->AllocMatrix(&cov3w1, pnl::matCovariance, -1, parent);
	pBNet->GetFactor(3)->AllocMatrix(&weight3w1, pnl::matWeights, 0, parent);

	// create DBN using BNet	
	pnl::CDBN *pArHMM = pnl::CDBN::Create(pBNet);

    return pArHMM;
}

void infer_mpe_in_hmm(pnl::CDBN *pHMM)
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
		evidences[time_slice] = pnl::CEvidence::Create(pHMM, obsNodes, obsNodesVals);
	}

	// create an inference engine
	boost::scoped_ptr<pnl::C1_5SliceJtreeInfEngine> inferEng(pnl::C1_5SliceJtreeInfEngine::Create(pHMM));

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
		
		std::cout << ">>> Query time-slice: " << time_slice << std::endl;

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
}

}  // namespace local
}  // unnamed namespace

void hmm()
{
	// simple HMM
	{
		boost::scoped_ptr<pnl::CDBN> simpleHMM(local::create_simple_hmm());
		//boost::scoped_ptr<pnl::CDBN> simpleHMM(local::create_hmm_with_ar_gaussian_observations());

		if (!simpleHMM)
		{
			std::cout << "can't create a probabilistic graphic model" << std::endl;
			return;
		}

		// get content of Graph
		simpleHMM->GetGraph()->Dump();
 
		if (false)
		{
			const pnl::CGraph *pGraph = simpleHMM->GetGraph();

			int numNbrs1, numNbrs2;
			const int *nbrs1, *nbrs2;
			const pnl::ENeighborType *nbrsTypes1, *nbrsTypes2;
			pGraph->GetNeighbors(0, &numNbrs1, &nbrs1, &nbrsTypes1);
			pGraph->GetNeighbors(1, &numNbrs2, &nbrs2, &nbrsTypes2);
		}

		local::infer_mpe_in_hmm(simpleHMM.get());
	}

	// HMM with AR Gaussian observations
	{
		boost::scoped_ptr<pnl::CDBN> arHMM(local::create_hmm_with_ar_gaussian_observations());

		if (!arHMM)
		{
			std::cout << "can't create a probabilistic graphical model" << std::endl;
			return;
		}

		// get content of Graph
		arHMM->GetGraph()->Dump();
	}
}
