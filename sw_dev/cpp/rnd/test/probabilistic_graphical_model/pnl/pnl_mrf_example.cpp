//#include "stdafx.h"
#include <pnl_dll.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>


namespace {
namespace local {

// [ref] testGetFactorsMRF2() in ${PNL_ROOT}/c_pgmtk/tests/src/AGetParametersTest.cpp
pnl::CMRF2 * create_simple_pairwise_mrf()
{
	const int numNodes = 7;
	const int numNodeTypes = 2;
	const int numCliques = 6;

#if 0
	const int cliqueSizes[] = { 2, 2, 2, 2, 2, 2 };

	const int clique0[] = { 0, 1 };
	const int clique1[] = { 1, 2 };
	const int clique2[] = { 1, 3 };
	const int clique3[] = { 2, 4 };
	const int clique4[] = { 2, 5 };
	const int clique5[] = { 3, 6 };

	const int *cliques[] = { clique0, clique1, clique2, clique3, clique4, clique5 };

	pnl::CNodeType *nodeTypes = new pnl::CNodeType [numNodeTypes];
	nodeTypes[0].SetType(true, 3);
	nodeTypes[1].SetType(true, 4);

	int *nodeAssociation = new int [numNodes];
	for (int i = 0; i < numNodes; ++i)
		nodeAssociation[i] = i % 2;

	pnl::CMRF2 *mrf2 = pnl::CMRF2::Create(numNodes, numNodeTypes, nodeTypes, nodeAssociation, numCliques, cliqueSizes, cliques);
#else
	pnl::intVecVector cliques;
	{
		const int clique0[] = { 0, 1 };
		const int clique1[] = { 1, 2 };
		const int clique2[] = { 1, 3 };
		const int clique3[] = { 2, 4 };
		const int clique4[] = { 2, 5 };
		const int clique5[] = { 3, 6 };
		cliques.push_back(pnl::intVector(clique0, clique0 + sizeof(clique0) / sizeof(clique0[0])));
		cliques.push_back(pnl::intVector(clique1, clique1 + sizeof(clique1) / sizeof(clique1[0])));
		cliques.push_back(pnl::intVector(clique2, clique2 + sizeof(clique2) / sizeof(clique2[0])));
		cliques.push_back(pnl::intVector(clique3, clique3 + sizeof(clique3) / sizeof(clique3[0])));
		cliques.push_back(pnl::intVector(clique4, clique4 + sizeof(clique4) / sizeof(clique4[0])));
		cliques.push_back(pnl::intVector(clique5, clique5 + sizeof(clique5) / sizeof(clique5[0])));
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

	pnl::CMRF2 *mrf2 = pnl::CMRF2::Create(numNodes, nodeTypes, nodeAssociation, cliques);
#endif

	mrf2->AllocFactors();
	for (int i = 0; i < numCliques; ++i)
		mrf2->AllocFactor(i);

	// get content of Graph
	mrf2->GetGraph()->Dump();

	//
	const int numQueries = 13;
	const int queryLength[] = { 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2 };
	const int queries[13][2] =
	{
		{ 0 },
		{ 1 },
		{ 2 },
		{ 3 },
		{ 4 },
		{ 5 },
		{ 6 },
		{ 0, 1 },
		{ 1, 3 },
		{ 1, 2 },
		{ 4, 2 },
		{ 2, 5 },
		{ 6, 3 }
	};

	pnl::pFactorVector params;
	for (int i = 0; i < numQueries; ++i)
	{
		mrf2->GetFactors(queryLength[i], queries[i], &params);

		// TODO [add] >>

		params.clear();
	}

#if 0
	delete [] nodeAssociation;
	delete [] nodeTypes;
#endif

	return mrf2;
}

}  // namespace local
}  // unnamed namespace

namespace my_pnl {

void mrf_example()
{
	// simple pairwise MRF
	std::cout << "========== simple pairwise MRF" << std::endl;
	{
		const boost::scoped_ptr<pnl::CMRF2> mrf2(local::create_simple_pairwise_mrf());

		if (!mrf2)
		{
			std::cout << "fail to create a probabilistic graphical model at " << __LINE__ << " in " << __FILE__ << std::endl;
			return;
		}
	}
}

}  // namespace my_pnl
