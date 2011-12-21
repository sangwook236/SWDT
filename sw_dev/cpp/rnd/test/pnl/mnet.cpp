#include "stdafx.h"
#include <pnl_dll.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>


namespace {
namespace local {

pnl::CMNet * create_simle_markov_network()
{
	const int numNodes = 7;
	const int numNodeTypes = 2;
	const int numCliques = 4;

#if 1
	const int cliqueSizes[] = { 3, 3, 3, 4 };

	const int clique0[] = { 1, 2, 3 };
	const int clique1[] = { 0, 1, 3 };
	const int clique2[] = { 4, 5, 6 };
	const int clique3[] = { 0, 3, 4, 5 };
	const int *cliques[] = { clique0, clique1, clique2, clique3 };

	pnl::CNodeType *nodeTypes = new pnl::CNodeType [numNodeTypes];
	nodeTypes[0].SetType(true, 3);
	nodeTypes[1].SetType(true, 4);

	int *nodeAssociation = new int [numNodes];
	for (int i = 0; i < numNodes; ++i)
		nodeAssociation[i] = i % 2;

    pnl::CMNet *mnet = pnl::CMNet::Create(numNodes, numNodeTypes, nodeTypes, nodeAssociation, numCliques, cliqueSizes, cliques);
#else
	pnl::intVecVector cliques;
	{
		const int clique0[] = { 1, 2, 3 };
		const int clique1[] = { 0, 1, 3 };
		const int clique2[] = { 4, 5, 6 };
		const int clique3[] = { 0, 3, 4, 5 };
		cliques.push_back(pnl::intVector(clique0, clique0 + sizeof(clique0) / sizeof(clique0[0])));
		cliques.push_back(pnl::intVector(clique1, clique1 + sizeof(clique1) / sizeof(clique1[0])));
		cliques.push_back(pnl::intVector(clique2, clique2 + sizeof(clique2) / sizeof(clique2[0])));
		cliques.push_back(pnl::intVector(clique3, clique3 + sizeof(clique3) / sizeof(clique3[0])));
	}

	//pnl::nodeTypeVector nodeTypes;
	//nodeTypes.push_back(pnl::CNodeType(true, 3));
	//nodeTypes.push_back(pnl::CNodeType(true, 4));
	pnl::nodeTypeVector nodeTypes(numNodeTypes);
	nodeTypes[0].SetType(true, 3);
	nodeTypes[1].SetType(true, 4);

	pnl::intVector nodeAssociation(numNodes);
	for (int i = 0; i < numNodes; ++i)
		nodeAssociation[i] = i % 2;

	pnl::CMNet *mnet = pnl::CMNet::Create(numNodes, nodeTypes, nodeAssociation, cliques);
#endif

	mnet->AllocFactors();
	for (int i = 0; i < numCliques; ++i)
		mnet->AllocFactor(i);

	// get content of Graph
	mnet->GetGraph()->Dump();

	//
	const int numQueries = 27;
	const int queryLength[] = { 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4 };
	const int queries[27][4] =
	{
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
		{ 4, 3 },
		{ 0, 4 },
		{ 5, 4 },
		{ 3, 5 },
		{ 6, 4 },
		{ 0, 6 },
		{ 6, 5 },
		{ 0, 5 },
		{ 0, 1, 3 },
		{ 3, 2, 1 },
		{ 5, 3, 4 },
		{ 6, 4, 5 },
		{ 0, 4, 3 },
		{ 0, 4, 5 },
		{ 4, 3, 5, 0 }
	};

	pnl::pFactorVector params;
	for (int i = 0; i < numQueries; ++i)
	{
		mnet->GetFactors(queryLength[i], queries[i], &params);

		// TODO [add] >>


		params.clear();
	}
	
#if 0
	delete [] nodeAssociation;
	delete [] nodeTypes;
#endif

	return mnet;
}

}  // namespace local
}  // unnamed namespace

void mnet()
{
	// simple markov network
	{
		boost::scoped_ptr<pnl::CMNet> mnet(local::create_simle_markov_network());
	}
}
