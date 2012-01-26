#include "stdafx.h"
#include <pnl_dll.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>


namespace {
namespace local {

pnl::CBNet * create_simle_bayesian_network()
{
	const int numNodes = 7;
	const int numNodeTypes = 2;

	const int numOfNeigh[] = { 2, 2, 2, 5, 3, 3, 3 };
	const int neigh0[] = { 3, 6 };
	const int neigh1[] = { 3, 4 };
	const int neigh2[] = { 3, 4 };
	const int neigh3[] = { 0, 1, 2, 5, 6 };
	const int neigh4[] = { 1, 2, 5 };
	const int neigh5[] = { 3, 4, 6 };
	const int neigh6[] = { 0, 3, 5 };
	const int *neigh[] = { neigh0, neigh1, neigh2, neigh3, neigh4, neigh5, neigh6 };

	const pnl::ENeighborType orient0[] = { pnl::ntChild, pnl::ntChild };
	const pnl::ENeighborType orient1[] = { pnl::ntChild, pnl::ntChild };
	const pnl::ENeighborType orient2[] = { pnl::ntChild, pnl::ntChild };
	const pnl::ENeighborType orient3[] = { pnl::ntParent, pnl::ntParent, pnl::ntParent, pnl::ntChild, pnl::ntChild };
	const pnl::ENeighborType orient4[] = { pnl::ntParent, pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient5[] = { pnl::ntParent, pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient6[] = { pnl::ntParent, pnl::ntParent, pnl::ntParent };
	const pnl::ENeighborType *orient[] = { orient0, orient1, orient2, orient3, orient4, orient5, orient6 };

	pnl::CGraph *graph = pnl::CGraph::Create(numNodes, numOfNeigh, neigh, orient);

	//
#if 0
	pnl::CNodeType *nodeTypes = new pnl::CNodeType [numNodeTypes];
	nodeTypes[0].SetType(true, 3);
	nodeTypes[1].SetType(true, 4);

	int *nodeAssociation = new int [numNodes];
	for (int i = 0; i < numNodes; ++i)
		nodeAssociation[i] = i % 2;

	pnl::CBNet *bnet = pnl::CBNet::Create(numNodes, numNodeTypes, nodeTypes, nodeAssociation, graph);
#else
	//pnl::nodeTypeVector nodeTypes;
	//nodeTypes.push_back(pnl::CNodeType(true, 3));
	//nodeTypes.push_back(pnl::CNodeType(true, 4));
	pnl::nodeTypeVector nodeTypes(numNodeTypes);
	nodeTypes[0].SetType(true, 3);
	nodeTypes[1].SetType(true, 4);

	pnl::intVector nodeAssociation(numNodes);
	for (int i = 0; i < numNodes; ++i)
		nodeAssociation[i] = i % 2;

	pnl::CBNet *bnet = pnl::CBNet::Create(numNodes, nodeTypes, nodeAssociation, graph);
#endif

	bnet->AllocFactors();
	for (int i = 0; i < numNodes; ++i)
		bnet->AllocFactor(i);

	// get content of Graph
	bnet->GetGraph()->Dump();

	//
	const int numQueries = 28;
	const int queryLength[] = { 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4 };
	const int queries[28][4] = {
		{ 0 },
		{ 1 },
		{ 2 },
		{ 3 },
		{ 4 },
		{ 5 },
		{ 6 },
		{ 0, 1 },
		{ 0, 3 },
		{ 1, 3 },
		{ 1, 2 },
		{ 2, 3 },
		{ 4, 1 },
		{ 2, 4 },
		{ 3, 4 },
		{ 3, 5 },
		{ 5, 4 },
		{ 0, 6 },
		{ 6, 5 },
		{ 6, 3 },
		{ 0, 2, 3 },
		{ 3, 2, 1 },
		{ 5, 3, 4 },
		{ 6, 3, 0 },
		{ 1, 2, 3 },
		{ 3, 2, 1 },
		{ 6, 3, 5, 0 },
		{ 3, 1, 2, 0 }
	};

	pnl::pFactorVector params;
	for (int i = 0; i < numQueries; ++i)
	{
		bnet->GetFactors(queryLength[i], queries[i], &params);

		// TODO [add] >>

		params.clear();
	}

#if 0
	delete [] nodeAssociation;
	delete [] nodeTypes;
#endif

	return bnet;
}

}  // namespace local
}  // unnamed namespace

void bnet()
{
	// simple Bayesina network
	{
		boost::scoped_ptr<pnl::CBNet> bnet(local::create_simle_bayesian_network());

		if (!bnet)
		{
			std::cout << "can't create a probabilistic graphical model" << std::endl;
			return;
		}
	}
}
