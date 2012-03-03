#if defined(_MSC_VER)
#include "stdafx.h"
#endif
#include <pnl_dll.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

// [ref]
//	"dHugin: A computational system for dynamic time-sliced Bayesian networks", U. Kjaerulff, Intl. J. Forecasting 11:89-111, 1995.
//	${PNL_ROOT}/c_pgmtk/tests/src/tCreateKjaerulffDBN.cpp
pnl::CDBN * create_Kjaerulff_dbn()
{
/*
	The intra structure is (all arcs point downwards)

		0 -> 1
		|  /
		V
		2
		|
		V
		3
		| \
		V   V
		4   5
		| /
		V
		6
		|
		V
		7

	The inter structure is 0->0, 3->3, 7->7.

	=================================================================

	1st slice              2nd slice

		0 -------------------> 8(1)
		|  \                   |  \
		v    v                 v    v
		2 <- 1                 9 -> 10
		|                      |
		v                      v
		3 -------------------> 11(3)
		| \                    | \
		v   v                  v   v
		4   5                  12  13
		| /                    | /
		v                      v
		6                      14
		|                      |
		v                      v
		7 -------------------> 15(7)

	where
		0, 1, 2, ..., 15 - bivariate tabular nodes
*/

	const int numNodes = 16;
	const int numNodeTypes = 1;

	const int numNeighs[] = {
		3, 2, 3, 4, 2, 2, 3, 2,  // 1st time-slice
		3, 2, 3, 4, 2, 2, 3, 2  // 2nd time-slice
	};

	const int neigh0[] = { 1, 2, 8 };
	const int neigh1[] = { 0, 2 };
	const int neigh2[] = { 0, 1, 3 };
	const int neigh3[] = { 2, 4, 5, 11 };
	const int neigh4[] = { 3, 6 };
	const int neigh5[] = { 3, 6 };
	const int neigh6[] = { 4, 5, 7 };
	const int neigh7[] = { 6, 15 };
	const int neigh8[] =  { 0, 9, 10 };
	const int neigh9[] =  { 8, 10 };
	const int neigh10[] = { 8, 9, 11 };
	const int neigh11[] = { 3, 10, 12, 13 };
	const int neigh12[] = { 11, 14 };
	const int neigh13[] = { 11, 14 };
	const int neigh14[] = { 12, 13, 15 };
	const int neigh15[] = { 7, 14 };
	const int *neighs[] = {
		neigh0, neigh1, neigh2, neigh3, neigh4, neigh5,
		neigh6, neigh7,	neigh8, neigh9, neigh10,
		neigh11, neigh12, neigh13, neigh14, neigh15
	};

	const pnl::ENeighborType orient0[] = { pnl::ntChild, pnl::ntChild, pnl::ntChild };
	const pnl::ENeighborType orient1[] = { pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient2[] = { pnl::ntParent, pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient3[] = { pnl::ntParent, pnl::ntChild, pnl::ntChild, pnl::ntChild };
	const pnl::ENeighborType orient4[] = { pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient5[] = { pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient6[] = { pnl::ntParent, pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient7[] = { pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient8[] = { pnl::ntParent, pnl::ntChild, pnl::ntChild };
	const pnl::ENeighborType orient9[] = { pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient10[] = { pnl::ntParent, pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient11[] = { pnl::ntParent, pnl::ntParent, pnl::ntChild, pnl::ntChild };
	const pnl::ENeighborType orient12[] = { pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient13[] = { pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient14[] = { pnl::ntParent, pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient15[] = { pnl::ntParent, pnl::ntParent };
	const pnl::ENeighborType *orients[] = {
		orient0, orient1, orient2, orient3, orient4, orient5,
		orient6, orient7, orient8, orient9, orient10,
		orient11, orient12, orient13, orient14, orient15
	};

	pnl::CGraph *graph = pnl::CGraph::Create(numNodes, numNeighs, neighs, orients);

	//
/*
	pnl::CNodeType *nodeTypes = new CNodeType [numNodeTypes];
	for (int i = 0; i < numNodeTypes; ++i)
	{
		nodeTypes[i] = pnl::CNodeType(true, 2);  // all nodes are discrete and binary
	}

	const int nodeAssociation[numNodes] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	pnl::CBNet *bnet = pnl::CBNet::Create(numNodes, numNodeTypes, nodeTypes, nodeAssociation, graph);
*/
	const pnl::nodeTypeVector nodeTypes(numNodeTypes, pnl::CNodeType(true, 2));
	const pnl::intVector nodeAssociation(numNodes, 0);

	pnl::CBNet *bnet = pnl::CBNet::Create(numNodes, nodeTypes, nodeAssociation, graph);

	//
	bnet->AllocFactors();
	for (int node = 0; node < numNodes; ++node)
	{
		bnet->AllocFactor(node);
		pnl::CFactor *factor = bnet->GetFactor(node);

		int domainSize;
		const int *domain;
		factor->GetDomain(&domainSize, &domain);

		int prodDomainSize = 1;
		for (int i = 0; i< domainSize; ++i)
		{
			prodDomainSize *= bnet->GetNodeType(domain[i])->GetNodeSize();
		}

		pnl::floatVector prior(prodDomainSize);
		for (int i = 0; i < prodDomainSize; ++i)
		{
			prior[i] = float(std::rand() % 10) + 1.0f;
		}

		factor->AllocMatrix(&prior.front(), pnl::matTable);
		dynamic_cast<pnl::CTabularCPD *>(factor)->NormalizeCPD();
	}

	// create DBN
	return pnl::CDBN::Create(bnet);
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/AJtreeInfMixtureDBN.cpp
pnl::CDBN * create_dbn_with_mixture_of_gaussians_observations()
{
	const int numNodes = 8;
	const int numNodeTypes = 2;

#if 0
/*
	the model #1 is

		       W(0) ------> W(4)
		       |            |
		       |            |
		       v            v
		       X(1) ------> X(5)
		     / |          / |
		    /  |         /  |
		   v   v        v   v
	    (2)Y-->Z(3)  (6)Y-->Z(7)

	where
		X - tabular node (bivariate)
		Z - Gaussian mixture node (univariate)
		Y - tabular nodes. It is a special node - mixture node (bivariate)
			it is used for storage summing coefficients for Gaussians.
			it must be a last discrete node among all discrete parents for Gaussian mixture node.
*/

	const int numNeighs[] = { 2, 4, 2, 2, 2, 4, 2, 2 };

	const int neigh0[] = { 1, 4 };
	const int neigh1[] = { 0, 2, 3, 5 };
	const int neigh2[] = { 1, 3 };
	const int neigh3[] = { 1, 2 };
	const int neigh4[] = { 0, 5 };
	const int neigh5[] = { 1, 4, 6, 7 };
	const int neigh6[] = { 5, 7 };
	const int neigh7[] = { 5, 6 };

	const pnl::ENeighborType orient0[] = { pnl::ntChild, pnl::ntChild };
	const pnl::ENeighborType orient1[] = { pnl::ntParent, pnl::ntChild, pnl::ntChild, pnl::ntChild };
	const pnl::ENeighborType orient2[] = { pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient3[] = { pnl::ntParent, pnl::ntParent };
	const pnl::ENeighborType orient4[] = { pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient5[] = { pnl::ntParent, pnl::ntParent, pnl::ntChild, pnl::ntChild };
	const pnl::ENeighborType orient6[] = { pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient7[] = { pnl::ntParent, pnl::ntParent };
#else
/*
	the model #2 is

		       W(0) ------> W(4)
		       |            |
		       |            |
		       v            v
		       X(1) ------> X(5)
		       |            |
		       |            |
		       v            v
	    (2)Y-->Z(3)  (6)Y-->Z(7)

	where
		X - tabular node (bivariate)
		Z - Gaussian mixture node (univariate)
		Y - tabular nodes. It is a special node - mixture node (bivariate)
			it is used for storage summing coefficients for Gaussians.
			it must be a last discrete node among all discrete parents for Gaussian mixture node.
*/

	const int numNeighs[] = { 2, 3, 1, 2, 2, 3, 1, 2 };

	const int neigh0[] = { 1, 4 };
	const int neigh1[] = { 0, 3, 5 };
	const int neigh2[] = { 3 };
	const int neigh3[] = { 1, 2 };
	const int neigh4[] = { 0, 5 };
	const int neigh5[] = { 1, 4, 7 };
	const int neigh6[] = { 7 };
	const int neigh7[] = { 5, 6 };

	const pnl::ENeighborType neighType0[] = { pnl::ntChild, pnl::ntChild };
	const pnl::ENeighborType neighType1[] = { pnl::ntParent, pnl::ntChild, pnl::ntChild };
	const pnl::ENeighborType neighType2[] = { pnl::ntChild };
	const pnl::ENeighborType neighType3[] = { pnl::ntParent, pnl::ntParent };
	const pnl::ENeighborType neighType4[] = { pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType neighType5[] = { pnl::ntParent, pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType neighType6[] = { pnl::ntChild };
	const pnl::ENeighborType neighType7[] = { pnl::ntParent, pnl::ntParent };
#endif
	const int *neighs[] = { neigh0, neigh1, neigh2, neigh3, neigh4, neigh5, neigh6, neigh7 };
	const pnl::ENeighborType *neighTypes[] = { neighType0, neighType1, neighType2, neighType3, neighType4, neighType5, neighType6, neighType7 };

	pnl::CGraph *graph = pnl::CGraph::Create(numNodes, numNeighs, neighs, neighTypes);

	//
/*
	pnl::CNodeType *nodeTypes = new pnl::CNodeType [numNodeTypes];
	nodeTypes[0] = pnl::CNodeType(true, 2);
	nodeTypes[1] = pnl::CNodeType(false, 1);

	int *nodeAssociation = new int [numNodes];
	nodeAssociation[0] = 0;
	nodeAssociation[1] = 0;
	nodeAssociation[2] = 0;
	nodeAssociation[3] = 1;
	nodeAssociation[4] = 0;
	nodeAssociation[5] = 0;
	nodeAssociation[6] = 0;
	nodeAssociation[7] = 1;

	pnl::CBNet *bnet = pnl::CBNet::Create(numNodes, numNodeTypes, nodeTypes, nodeAssociation, graph);

	delete [] nodeTypes;
	nodeTypes = NULL;
	delete [] nodeAssociation;
	nodeAssociation = NULL;
*/
	pnl::nodeTypeVector nodeTypes(numNodeTypes);
	nodeTypes[0].SetType(true, 2);
	nodeTypes[1].SetType(false, 1);

	pnl::intVector nodeAssociation(numNodes, 0);  // { 0, 0, 0, 1, 0, 0, 0, 1 }
	nodeAssociation[3] = 1;
	nodeAssociation[7] = 1;

	pnl::CBNet *bnet = pnl::CBNet::Create(numNodes, nodeTypes, nodeAssociation, graph);

	// create domains
#if 0
	// model #1
	const int domain0[] = { 0 };
	const int domain1[] = { 0, 1 };
	const int domain2[] = { 1, 2 };
	const int domain3[] = { 1, 2, 3 };
	const int domain4[] = { 0, 4 };
	const int domain5[] = { 1, 4, 5 };
	const int domain6[] = { 5, 6 };
	const int domain7[] = { 5, 6, 7 };
	const int *domains[] = { domain0, domain1, domain2, domain3, domain4, domain5, domain6, domain7 };

	std::vector<int> nodeNumbers(numNodes, 0);
	nodeNumbers[0] = 1;
	nodeNumbers[1] = 2;
	nodeNumbers[2] = 2;
	nodeNumbers[3] = 3;
	nodeNumbers[4] = 2;
	nodeNumbers[5] = 3;
	nodeNumbers[6] = 2;
	nodeNumbers[7] = 3;
#else
	// model #2
	const int domain0[] = { 0 };
	const int domain1[] = { 0, 1 };
	const int domain2[] = { 2 };
	const int domain3[] = { 1, 2, 3 };
	const int domain4[] = { 0, 4 };
	const int domain5[] = { 1, 4, 5 };
	const int domain6[] = { 6 };
	const int domain7[] = { 5, 6, 7 };
	const int *domains[] = { domain0, domain1, domain2, domain3, domain4, domain5, domain6, domain7 };

	std::vector<int> nodeNumbers(numNodes, 0);
	nodeNumbers[0] = 1;
	nodeNumbers[1] = 2;
	nodeNumbers[2] = 1;
	nodeNumbers[3] = 3;
	nodeNumbers[4] = 2;
	nodeNumbers[5] = 3;
	nodeNumbers[6] = 1;
	nodeNumbers[7] = 3;
#endif

	pnl::pnlVector<pnl::pConstNodeTypeVector> nt(numNodes, pnl::pConstNodeTypeVector());
	for (int i = 0; i < numNodes; ++i)
	{
		const int size = nodeNumbers[i];

		for (int j = 0; j < size; ++j)
			nt[i].push_back(bnet->GetNodeType(domains[i][j]));
	}

	bnet->AllocFactors();

	//
	const float table0[] = { 0.7f, 0.3f };  // node X
	const float table1[] = { 0.79f, 0.21f, 0.65f, 0.35f };
	const float table2[] = { 0.5f, 0.5f, 0.5f, 0.5f };
	const float table3[] = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
	//const float table1[] = { 0.1f, 0.9f };  // node Y

	const float mean100 = 1.0f, cov100 = 0.5f;  // node Z for X = 0, Y = 0
	const float mean110 = -5.0f, cov110 = 4.0f;  // node Z for X = 1, Y = 0
	const float mean101 = -3.0f, cov101 = 1.5f;  // node Z for X = 0, Y = 1
	const float mean111 = 2.0f, cov111 = 1.0f;  // node Z for X = 1, Y = 1

	pnl::CTabularCPD *cpd0 = pnl::CTabularCPD::Create(domains[0], nodeNumbers[0], bnet->GetModelDomain(), table0);
	bnet->AttachFactor(cpd0);

	pnl::CTabularCPD *cpd1 = pnl::CTabularCPD::Create(domains[1], nodeNumbers[1], bnet->GetModelDomain(), table1);
	bnet->AttachFactor(cpd1);

	pnl::CTabularCPD *cpd2 = pnl::CTabularCPD::Create(domains[2], nodeNumbers[2], bnet->GetModelDomain(), table1);
	bnet->AttachFactor(cpd2);

	pnl::CMixtureGaussianCPD *cpd3 = pnl::CMixtureGaussianCPD::Create(domains[3], nodeNumbers[3], bnet->GetModelDomain(), table1);
	pnl::intVector parentVal(2, 0);
	cpd3->AllocDistribution(&mean100, &cov100, 2.0f, NULL, &parentVal.front());
	parentVal[1] = 1;
	cpd3->AllocDistribution(&mean101, &cov101, 1.0f, NULL, &parentVal.front());
	parentVal[0] = 1;
	cpd3->AllocDistribution(&mean111, &cov111, 1.0f, NULL, &parentVal.front());
	parentVal[1] = 0;
	cpd3->AllocDistribution(&mean110, &cov110, 1.0f, NULL, &parentVal.front());
	bnet->AttachFactor(cpd3);

	pnl::CTabularCPD *cpd4 = pnl::CTabularCPD::Create(domains[4], nodeNumbers[4], bnet->GetModelDomain(), table2);
	bnet->AttachFactor(cpd4);

	pnl::CTabularCPD *cpd5 = pnl::CTabularCPD::Create(domains[5], nodeNumbers[5], bnet->GetModelDomain(), table3);
	bnet->AttachFactor(cpd5);

	pnl::CTabularCPD *cpd6 = pnl::CTabularCPD::Create(domains[6], nodeNumbers[6], bnet->GetModelDomain(), table1);
	bnet->AttachFactor(cpd6);

	pnl::CMixtureGaussianCPD *cpd7 = pnl::CMixtureGaussianCPD::Create(domains[7], nodeNumbers[7], bnet->GetModelDomain(), table1);
	parentVal.assign(2, 0);
	cpd7->AllocDistribution(&mean100, &cov100, 2.0f, NULL, &parentVal.front());
	parentVal[1] = 1;
	cpd7->AllocDistribution(&mean101, &cov101, 1.0f, NULL, &parentVal.front());
	parentVal[0] = 1;
	cpd7->AllocDistribution(&mean111, &cov111, 1.0f, NULL, &parentVal.front());
	parentVal[1] = 0;
	cpd7->AllocDistribution(&mean110, &cov110, 1.0f, NULL, &parentVal.front());
	bnet->AttachFactor(cpd7);

	//
	return pnl::CDBN::Create(bnet);
}

// [ref]
//	${PNL_ROOT}/c_pgmtk/tests/src/AJtreeInfMixtureDBN.cpp
//	${PNL_ROOT}/c_pgmtk/tests/src/A1_5JTreeInfDBNCondGauss.cpp
void infer_dbn_with_mixture_of_gaussians_observations_using_1_5_junction_tree_inference_algorithm(const boost::scoped_ptr<pnl::CDBN> &dbn)
{
	const int numTimeSlices = 4;
	const int numNodes = dbn->GetStaticModel()->GetNumberOfNodes();  // TODO [check] >> does it have a correct value?

	const boost::scoped_ptr<const pnl::CBNet> unrolledBNet(static_cast<pnl::CBNet *>(dbn->UnrollDynamicModel(numTimeSlices)));

	// create evidence for every slice
	pnl::pEvidencesVector evidences(numTimeSlices);

	// node 3 is always observed
	const pnl::intVector obsNodeNums(1, 3);
	pnl::valueVector obsVals(1);
	pnl::intVector obsNodeNumsForUnrolled(numTimeSlices);
	pnl::valueVector obsValsForUnrolled(numTimeSlices);

	for (int i = 0; i < numTimeSlices; ++i)
	{
		const float ft = std::rand() / 10.0f;

		obsVals[0].SetFlt(ft);
		evidences[i] = pnl::CEvidence::Create(dbn->GetModelDomain(), obsNodeNums, obsVals);

		obsValsForUnrolled[i].SetFlt(ft);
		obsNodeNumsForUnrolled[i] = obsNodeNums[0] + numNodes / 2 * i;
	}

	const boost::scoped_ptr<pnl::CEvidence> evidencesForUnrolled(pnl::CEvidence::Create(unrolledBNet->GetModelDomain(), obsNodeNumsForUnrolled, obsValsForUnrolled));

	//
	const boost::scoped_ptr<pnl::C1_5SliceJtreeInfEngine> infEngine(pnl::C1_5SliceJtreeInfEngine::Create(dbn.get()));
	infEngine->DefineProcedure(pnl::ptSmoothing, numTimeSlices);
	infEngine->EnterEvidence(&evidences.front(), numTimeSlices);
	infEngine->Smoothing();

	//
	const boost::scoped_ptr<pnl::CNaiveInfEngine> infEngineForUnrolled(pnl::CNaiveInfEngine::Create(unrolledBNet.get()));
	infEngineForUnrolled->EnterEvidence(evidencesForUnrolled.get());

	//
	{
		// node 0 & 1 ==> hidden states (for the prior network)
		pnl::intVector query(2, 0);
		pnl::intVector queryForUnrolled(2, 0);
		query[1] = 1;
		queryForUnrolled[1] = 1;

		infEngine->MarginalNodes(&query.front(), query.size(), 0);

		std::cout << " #-- initial model: time-slice 0" << std::endl;
		infEngine->GetQueryJPD()->Dump();

		infEngineForUnrolled->MarginalNodes(&queryForUnrolled.front(), queryForUnrolled.size());

		std::cout << " #-- unrolled model: time-slice 0" << std::endl;
		infEngineForUnrolled->GetQueryJPD()->Dump();

		// node 4 & 5 ==> hidden states (for the transition network)
		query[0] = numNodes / 2;
		query[1] = numNodes / 2 + 1;
		for (int t = 1; t < numTimeSlices; ++t)
		{
			infEngine->MarginalNodes(&query.front(), query.size(), t);

			std::cout << " #-- initial model: time-slice " << t << std::endl;
			infEngine->GetQueryJPD()->Dump();

			queryForUnrolled[0] += numNodes / 2;
			queryForUnrolled[1] += numNodes / 2;
			infEngineForUnrolled->MarginalNodes(&queryForUnrolled.front(), queryForUnrolled.size());

			std::cout << " #-- unrolled model: time-slice " << t << std::endl;
			infEngineForUnrolled->GetQueryJPD()->Dump();
		}
	}

	//
	for (int i = 0; i < numTimeSlices; ++i)
	{
		delete evidences[i];
	}
}

// [ref]
//	${PNL_ROOT}/c_pgmtk/tests/src/ABKInfDBN.cpp
//	${PNL_ROOT}/c_pgmtk/tests/src/ABKInfUsingClusters.cpp
void infer_dbn_with_mixture_of_gaussians_observations_using_boyen_koller_inference_algorithm(const boost::scoped_ptr<pnl::CDBN> &hmm)
{
	throw std::runtime_error("not yet implemented");
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/A2TPFInfDBN.cpp
void infer_dbn_with_mixture_of_gaussians_observations_using_2T_slice_particle_filtering_inference_algorithm(const boost::scoped_ptr<pnl::CDBN> &hmm)
{
	throw std::runtime_error("not yet implemented");
}

}  // namespace local
}  // unnamed namespace

void dbn()
{
	// DBN with mixture-of-Gaussians observations
	std::cout << "========== infer DBN with mixture-of-Gaussians observations" << std::endl;
	{
		const boost::scoped_ptr<pnl::CDBN> dbn(local::create_dbn_with_mixture_of_gaussians_observations());
		//const boost::scoped_ptr<pnl::CDBN> dbn(local::create_Kjaerulff_dbn());

		if (!dbn)
		{
			std::cout << "fail to create a probabilistic graphical model at " << __LINE__ << " in " << __FILE__ << std::endl;
			return;
		}

		// 1.5 slice junction tree inference algorithm
		local::infer_dbn_with_mixture_of_gaussians_observations_using_1_5_junction_tree_inference_algorithm(dbn);
		// Boyen-Koller (BK) inference algorithm (approximate algorithm)
		//local::infer_dbn_with_mixture_of_gaussians_observations_using_boyen_koller_inference_algorithm(dbn);  // not yet implemented
		// 2T slice particle filtering inference algorithm (approximate algorithm)
		//local::infer_dbn_with_mixture_of_gaussians_observations_using_2T_slice_particle_filtering_inference_algorithm(dbn);  // not yet implemented
	}
}
