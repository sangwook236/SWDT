#include "stdafx.h"
#include <pnl_dll.hpp>
#include <iostream>


namespace {
namespace local {

pnl::CBNet * create_model()
{
	const int numNodes = 4;

	// 1 STEP:
	// need to specify the graph structure of the model;
	// there are two way to do it
	pnl::CGraph *pGraph = NULL;
	if (true)
	{
		// graph creation using adjacency matrix

		const int numAdjMatDims = 2;

		const int ranges[] = { numNodes, numNodes };
		pnl::intVector matrixData(numNodes * numNodes, 0);
		pnl::CDenseMatrix<int> *adjMat = pnl::CDenseMatrix<int>::Create(numAdjMatDims, ranges, &matrixData.front());

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
		pGraph = pnl::CGraph::Create(adjMat);

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
		pGraph = pnl::CGraph::Create(numNodes, numOfNbrs, nbrs, nbrsTypes);
	}

	// 2 STEP:
	// creation NodeType objects and specify node types for all nodes of the model.
	pnl::nodeTypeVector nodeTypes;

	// number of node types is 1, because all nodes are of the same type
	// all four are discrete and binary
	pnl::CNodeType nt(1, 2);
	nodeTypes.push_back(nt);

	pnl::intVector nodeAssociation;
	// reflects association between node numbers and node types
	// nodeAssociation[k] is a number of node type object in the node types array for the k-th node
	nodeAssociation.assign(numNodes, 0);

	// 2 STEP:
	// create base for BNet using Graph, types of nodes and nodes association
	pnl::CBNet *pBNet = pnl::CBNet::Create(numNodes, nodeTypes, nodeAssociation, pGraph);

	// 3 STEP:
	// allocation space for all factors of the model
	pBNet->AllocFactors();

	// 4 STEP:
	// create factors and attach their to model

	// create raw data tables for CPDs
	const float table0[] = { 0.5f, 0.5f };
	const float table1[] = { 0.5f, 0.5f, 0.9f, 0.1f };
	const float table2[] = { 0.8f, 0.2f, 0.2f, 0.8f };
	const float table3[] = { 1.0f, 0.0f, 0.1f, 0.9f, 0.1f, 0.9f, 0.01f, 0.99f };
	const float *table[] = { table0, table1, table2, table3 };

	for (int i = 0; i < numNodes; ++i)
	{
		pBNet->AllocFactor(i);
		pnl::CFactor *pFactor = pBNet->GetFactor(i);
		pFactor->AllocMatrix(table[i], pnl::matTable);
	}

	return pBNet;
}

void infer_model()
{
	// create Water-Sprinkler BNet
	pnl::CBNet *pWSBnet = pnl::pnlExCreateWaterSprinklerBNet();
	//pnl::CBNet *pWSBnet = create_model();

	// get content of Graph
	pWSBnet->GetGraph()->Dump();

	// create simple evidence for node 0 from BNet
	pnl::CEvidence *pEvidForWS = 0L;
	{
		// make one node observed
		int nObsNds = 1;
		// the observed node is 0
		int obsNds[] = { 0 };
		// node 0 takes its second value (from two possible values {0, 1})
		pnl::valueVector obsVals;
		obsVals.resize(1);
		obsVals[0].SetInt(1);
		pEvidForWS = pnl::CEvidence::Create(pWSBnet, nObsNds, obsNds, obsVals);
	}

	// create Naive inference for BNet
	pnl::CNaiveInfEngine *pNaiveInf = pnl::CNaiveInfEngine::Create(pWSBnet);

	// enter evidence created before
	pNaiveInf->EnterEvidence(pEvidForWS);

	// get a marginal for query set of nodes
	int numQueryNds = 2;
	int queryNds[] = { 1, 3 };

	pNaiveInf->MarginalNodes(queryNds, numQueryNds);
	const pnl::CPotential *pMarg = pNaiveInf->GetQueryJPD();

	{
		pnl::intVector obsNds;
		pnl::pConstValueVector obsVls;
		pEvidForWS->GetObsNodesWithValues(&obsNds, &obsVls);

		for (int i = 0; i < obsNds.size(); ++i)
		{
			std::cout << " observed value for node " << obsNds[i];
			std::cout << " is " << obsVls[i]->GetInt() << std::endl;
		}
	}

	int nnodes;
	const int *domain = 0L;
	pMarg->GetDomain(&nnodes, &domain);
	std::cout << " inference results: " << std::endl;
	std::cout << " probability distribution for nodes [ ";
	for (int i = 0; i < nnodes; ++i)
	{
		std::cout << domain[i] << " ";
	}
	std::cout << "]" << std::endl;

	pnl::CMatrix<float> *pMat = pMarg->GetMatrix(pnl::matTable);

	// graphical model hase been created using dense matrix
	// so, the marginal is also dense
	pnl::EMatrixClass type = pMat->GetMatrixClass();
	if (!(type == pnl::mcDense || type == pnl::mcNumericDense || type == pnl::mc2DNumericDense))
	{
		assert(0);
	}

	int nEl;
	const float *data = NULL;
	static_cast<pnl::CNumericDenseMatrix<float> *>(pMat)->GetRawData(&nEl, &data);
	for (int i = 0; i < nEl; ++i)
	{
		std::cout << " " << data[i];
	}
	std::cout << std::endl;

	delete pEvidForWS;
	delete pNaiveInf;
	delete pWSBnet;
}

}  // namespace local
}  // unnamed namespace

void bayesian_network()
{
	local::infer_model();
}
