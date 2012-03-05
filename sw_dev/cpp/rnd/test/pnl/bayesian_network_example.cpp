//#include "stdafx.h"
#include <pnl_dll.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>
#include <stdexcept>

namespace {
namespace local {

pnl::CBNet * create_model()
{
	const int numNodes = 4;
	const int numNodeTypes = 1;

	// 1 STEP:
	// need to specify the graph structure of the model;
	// there are two way to do it
	pnl::CGraph *graph = NULL;
	if (true)
	{
		// graph creation using adjacency matrix

		const int numAdjMatDims = 2;

		const int ranges[] = { numNodes, numNodes };
		pnl::intVector matrixData(numNodes * numNodes, 0);
		pnl::CDenseMatrix<int> *adjMat = pnl::CDenseMatrix<int>::Create(numAdjMatDims, ranges, &matrixData.front());

		// assign indices of child nodes to each node
		int indices[] = { 0, 1 };
		adjMat->SetElementByIndexes(1, indices);
		indices[1] = 2;
		adjMat->SetElementByIndexes(1, indices);
		indices[0] = 1;
		indices[1] = 3;
		adjMat->SetElementByIndexes(1, indices);
		indices[0] = 2;
		adjMat->SetElementByIndexes(1, indices);

		// this is a creation of directed graph for the BNet model based on adjacency matrix
		graph = pnl::CGraph::Create(adjMat);

		delete adjMat;
	}
	else
	{
		// graph creation using neighbors list

		const int numOfNbrs[numNodes] = { 2, 2, 2, 2 };
		const int nbrs0[] = { 1, 2 };
		const int nbrs1[] = { 0, 3 };
		const int nbrs2[] = { 0, 3 };
		const int nbrs3[] = { 1, 2 };

		// number of neighbors for every node
		const int *nbrs[] = { nbrs0, nbrs1, nbrs2, nbrs3 };

		// neighbors can be of either one of the three following types:
		// a parent, a child (for directed arcs) or just a neighbor (for undirected graphs).
		// accordingly, the types are ntParent, ntChild or ntNeighbor.
		const pnl::ENeighborType nbrsTypes0[] = { pnl::ntChild, pnl::ntChild };
		const pnl::ENeighborType nbrsTypes1[] = { pnl::ntParent, pnl::ntChild };
		const pnl::ENeighborType nbrsTypes2[] = { pnl::ntParent, pnl::ntChild };
		const pnl::ENeighborType nbrsTypes3[] = { pnl::ntParent, pnl::ntParent };
		const pnl::ENeighborType *nbrsTypes[] = { nbrsTypes0, nbrsTypes1, nbrsTypes2, nbrsTypes3 };

		// this is creation of a directed graph for the BNet model using neighbors list
		graph = pnl::CGraph::Create(numNodes, numOfNbrs, nbrs, nbrsTypes);
	}

	// 2 STEP:
	// creation NodeType objects and specify node types for all nodes of the model.
	const pnl::nodeTypeVector nodeTypes(numNodeTypes, pnl::CNodeType(true, 2));
	// number of node types is 1, because all nodes are of the same type
	// all four are discrete and binary

	const pnl::intVector nodeAssociation(numNodes, 0);
	// reflects association between node numbers and node types
	// nodeAssociation[k] is a number of node type object in the node types array for the k-th node

	// 2 STEP:
	// create base for BNet using Graph, types of nodes and nodes association
	pnl::CBNet *bnet = pnl::CBNet::Create(numNodes, nodeTypes, nodeAssociation, graph);

	// 3 STEP:
	// allocation space for all factors of the model
	bnet->AllocFactors();

	// 4 STEP:
	// create factors and attach their to model

	// create raw data tables for CPDs
	const float table0[] = { 0.5f, 0.5f };
	const float table1[] = { 0.5f, 0.5f, 0.9f, 0.1f };
	const float table2[] = { 0.8f, 0.2f, 0.2f, 0.8f };
	const float table3[] = { 1.0f, 0.0f, 0.1f, 0.9f, 0.1f, 0.9f, 0.01f, 0.99f };
	const float *tables[] = { table0, table1, table2, table3 };

	for (int i = 0; i < numNodes; ++i)
	{
		bnet->AllocFactor(i);
		pnl::CFactor *factor = bnet->GetFactor(i);
		factor->AllocMatrix(tables[i], pnl::matTable);
	}

	return bnet;
}

void infer_bayesian_network_using_naive_inference_algorithm(const boost::scoped_ptr<pnl::CBNet> &bnet)
{
	// create simple evidence for node 0 from BNet
	pnl::CEvidence *evidForWS = NULL;
	{
		// make one node observed
		const int numObsNodes = 1;
		// the observed node is 0
		const int obsNodes[] = { 0 };
		// node 0 takes its second value (from two possible values {0, 1})
		pnl::valueVector obsVals(numObsNodes);
		obsVals[0].SetInt(1);

		evidForWS = pnl::CEvidence::Create(bnet.get(), numObsNodes, obsNodes, obsVals);
	}

	// create Naive inference for BNet
	pnl::CNaiveInfEngine *naiveInFEngine = pnl::CNaiveInfEngine::Create(bnet.get());

	// enter evidence created before
	naiveInFEngine->EnterEvidence(evidForWS);

	// get a marginal for query set of nodes
	const int numQueryNodes = 2;
	const int queryNodes[] = { 1, 3 };

	naiveInFEngine->MarginalNodes(queryNodes, numQueryNodes);
	const pnl::CPotential *jpd = naiveInFEngine->GetQueryJPD();

	{
		pnl::intVector obsNodes;
		pnl::pConstValueVector obsVals;
		evidForWS->GetObsNodesWithValues(&obsNodes, &obsVals);

		for (size_t i = 0; i < obsNodes.size(); ++i)
		{
			std::cout << " observed value for node " << obsNodes[i];
			std::cout << " is " << obsVals[i]->GetInt() << std::endl;
		}

		int nnodes;
		const int *domain = NULL;
		jpd->GetDomain(&nnodes, &domain);
		std::cout << " inference results: " << std::endl;
		std::cout << " probability distribution for nodes [ ";
		for (int i = 0; i < nnodes; ++i)
		{
			std::cout << domain[i] << " ";
		}
		std::cout << "]" << std::endl;
	}

	{
		pnl::CMatrix<float> *jpdMat = jpd->GetMatrix(pnl::matTable);

		// graphical model hase been created using dense matrix
		// so, the marginal is also dense
		const pnl::EMatrixClass type = jpdMat->GetMatrixClass();
		if (!(type == pnl::mcDense || type == pnl::mcNumericDense || type == pnl::mc2DNumericDense))
		{
			assert(0);
		}

		int numElem;
		const float *data = NULL;
		static_cast<pnl::CNumericDenseMatrix<float> *>(jpdMat)->GetRawData(&numElem, &data);
		for (int i = 0; i < numElem; ++i)
		{
			std::cout << " " << data[i];
		}
		std::cout << std::endl;
	}

	delete naiveInFEngine;
	delete evidForWS;
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/APearlInfEngine.cpp
void infer_bayesian_network_using_belief_propagation_algorithm(const boost::scoped_ptr<pnl::CBNet> &bnet)
{
	throw std::runtime_error("not yet implemented");
}

}  // namespace local
}  // unnamed namespace

void bayesian_network_example()
{
	std::cout << "========== infer water-sprinkler Bayesian network" << std::endl;
	{
		// create Water-Sprinkler BNet
#if 1
		const boost::scoped_ptr<pnl::CBNet> wsBNet(pnl::pnlExCreateWaterSprinklerBNet());
#else
		const boost::scoped_ptr<pnl::CBNet> wsBNet(local::create_model());
#endif

		if (!wsBNet)
		{
			std::cout << "fail to create a probabilistic graphical model at " << __LINE__ << " in " << __FILE__ << std::endl;
			return;
		}

		// get content of Graph
		wsBNet->GetGraph()->Dump();

		// naive inference algorithm
		local::infer_bayesian_network_using_naive_inference_algorithm(wsBNet);
		// belief propagation (Pearl inference) algorithm
		//local::infer_bayesian_network_using_belief_propagation_algorithm(wsBNet);  // not yet implemented
	}
}
