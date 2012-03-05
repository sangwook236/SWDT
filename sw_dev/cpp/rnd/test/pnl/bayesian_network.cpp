//#include "stdafx.h"
#include <pnl_dll.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

pnl::CBNet * create_discrete_bayesian_network()
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

	// get content of graph
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

// [ref] pnlExCreateSingleGauMix() in ${PNL_ROOT}/c_pgmtk/src/pnlExampleModels.cpp
pnl::CBNet * create_single_mixture_of_gaussians_bayesian_network()
{
/*
	the model

      Y

		Y - Gaussain mixture node

	In PNL graph for this model

	  0 -> 1

	where
		0 - tabular nodes. It is a special node - mixture node.
			it is used for storage summing coefficients for Gaussians.
			it must be a last discrete node among all discrete parents for Gaussian mixture node.
		1 - Gaussain mixture node. (univariate)
		nodes 0 & 1 together correspond to node Y.
*/

	// first need to specify the graph structure of the model/
	const int numNodes = 2;

#if 0
	const int numNeighbors[numNodes] = { 1, 1 };

	const int nbrs0[] = { 1 };
	const int nbrs1[] = { 0 };
	const int *nbrs[] = { nbrs0, nbrs1 };

	// neighbors can be of either one of three following types:
	// a parent, a child or just a neighbor - for undirected graphs.
	// if a neighbor of a node is it's parent, then neighbor type is pnl::ntParent
	// if it's a child, then pnl::ntChild and if it's a neighbor, then pnl::ntNeighbor
	const pnl::ENeighborType nbrsTypes0[] = { pnl::ntChild };
	const pnl::ENeighborType nbrsTypes1[] = { pnl::ntParent };
	const pnl::ENeighborType *nbrsTypes[] = { nbrsTypes0, nbrsTypes1 };

	pnl::CGraph *graph = pnl::CGraph::Create(numNodes, numNeighbors, nbrs, nbrsTypes);
#else
	pnl::intVecVector nbrs(numNodes);
	{
		pnl::intVector nbr(1);
		nbr[0] = 1;
		nbrs[0] = nbr;
		nbr[0] = 0;
		nbrs[1] = nbr;
	}

	pnl::neighborTypeVecVector nbrsTypes(numNodes);
	{
		pnl::neighborTypeVector nbrType(1);
		nbrType[0] = pnl::ntChild;
		nbrsTypes[0] = nbrType;
		nbrType[0] = pnl::ntParent;
		nbrsTypes[1] = nbrType;
	}

	pnl::CGraph *graph = pnl::CGraph::Create(nbrs, nbrsTypes);
#endif

	// number of node types is 1, because all nodes are of the same type
	// all four are discrete and binary
	const int numNodeTypes = 2;
	pnl::nodeTypeVector nodeTypes(numNodeTypes);
	nodeTypes[0].SetType(true, 2);  // discrete & binary
	nodeTypes[1].SetType(false, 1);  // continuous & univariate

	pnl::intVector nodeAssociation(numNodes);
	nodeAssociation[0] = 0;
	nodeAssociation[1] = 1;

	pnl::CBNet *bnet = pnl::CBNet::Create(numNodes, nodeTypes, nodeAssociation, graph);
	pnl::CModelDomain *md = bnet->GetModelDomain();

	// set factors
	bnet->AllocFactors();

	// create arrays of data for factors
	const float mixCoeffTable[] = { 0.7f, 0.3f };  // node 0
	const float mean0 = -1.0f, cov0 = 0.02f;  // node 1 for node 0 = 0
	const float mean1 =  5.0f, cov1 = 0.01f;  // node 1 for node 0 = 1

	// create domains
	const int node0Size = 1;
	const int node1Size = 2;
	const int nodeSizes[] = { node0Size, node1Size };

	const int domain0[] = { 0 };
	const int domain1[] = { 0, 1 };
	const int *domains[] = { domain0, domain1 };

	pnl::CTabularCPD *cpd0 = pnl::CTabularCPD::Create(domains[0], nodeSizes[0], md, mixCoeffTable);
	bnet->AttachFactor(cpd0);

	// the last argument must be equal to table for node 0, because node 0 (mixture node) is used for storage summing coefficients for Gaussians
	pnl::CMixtureGaussianCPD *cpd1 = pnl::CMixtureGaussianCPD::Create(domains[1], nodeSizes[1], md, mixCoeffTable);

	int parentVal = 0;
	cpd1->AllocDistribution(&mean0, &cov0, 2.0f, NULL, &parentVal);

	parentVal = 1;
	cpd1->AllocDistribution(&mean1, &cov1, 1.0f, NULL, &parentVal);
	bnet->AttachFactor(cpd1);

	return bnet;
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/AMixtureGaussainLearning.cpp
void learn_single_mixture_of_gaussians_bayesian_network(const boost::scoped_ptr<pnl::CBNet> &mogBNet)
{
	// FIXME [fix] >> run-time error

	// create data for learning
	const int numEvidences = 100;

	pnl::pEvidencesVector evidences;
	mogBNet->GenerateSamples(&evidences, numEvidences);

	// learn single mixture-of-Gaussians BNet
	boost::scoped_ptr<pnl::CBNet> mogBNetToLearn(pnl::CBNet::Copy(mogBNet.get()));

#if 0
	boost::scoped_ptr<pnl::CEMLearningEngine> learnEngine(pnl::CEMLearningEngine::Create(mogBNetToLearn.get()));
#else
	boost::scoped_ptr<pnl::CJtreeInfEngine> juncTreeInfEngine(pnl::CJtreeInfEngine::Create(mogBNet.get()));
	boost::scoped_ptr<pnl::CEMLearningEngine> learnEngine(pnl::CEMLearningEngine::Create(mogBNetToLearn.get(), juncTreeInfEngine.get()));
#endif
	learnEngine->SetData(numEvidences, &evidences.front());
	try
	{
		learnEngine->Learn();
	}
	catch (const pnl::CAlgorithmicException &e)
	{
		std::cout << "fail to learn parameters of a single mixture-of-Gaussians Bayesian Network" << e.GetMessage() << std::endl;
		return;
	}

	//
	const int numNodes = mogBNet->GetNumberOfNodes();

	const float eps = 1.5e-1f;
	for (int i = 0; i < numNodes; ++i)
	{
		if (!mogBNet->GetFactor(i)->IsFactorsDistribFunEqual(mogBNetToLearn->GetFactor(i), eps, 0))
		{
			std::cout << "original model & learned model are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;

			mogBNet->GetFactor(i)->GetDistribFun()->Dump();
			mogBNetToLearn->GetFactor(i)->GetDistribFun()->Dump();
		}
	}

	//
	for (int i = 0; i < numEvidences; ++i)
	{
		delete evidences[i];
	}
	evidences.clear();
}

void infer_single_mixture_of_gaussians_bayesian_network(const boost::scoped_ptr<pnl::CBNet> &mogBNet)
{
	// FIXME [fix] >> run-time error

	//const int numNodes = mogBNet->GetNumberOfNodes();

	// create evidence on all Gaussian nodes for inference
	const int numObsNodes = 1;
	const int obsNodes[] = { 1 };

	pnl::valueVector obsVals(numObsNodes, pnl::Value(0));
	obsVals[0].SetFlt(2.0f);

	boost::scoped_ptr<pnl::CEvidence> evidence(pnl::CEvidence::Create(mogBNet.get(), numObsNodes, obsNodes, obsVals));

	// create inference engine
	boost::scoped_ptr<pnl::CNaiveInfEngine> naiveInfEngine(pnl::CNaiveInfEngine::Create(mogBNet.get()));
	boost::scoped_ptr<pnl::CJtreeInfEngine> juncTreeInfEngine(pnl::CJtreeInfEngine::Create(mogBNet.get()));

	// start inference with maximization
	{
		const int maximizeFlag = 1;
		const int queryNode = 0;

		// naive inference
		naiveInfEngine->EnterEvidence(evidence.get(), maximizeFlag);
		naiveInfEngine->MarginalNodes(&queryNode, 1);
		const pnl::CEvidence *mpeEvidNaive = naiveInfEngine->GetMPE();
		const int mpeValNaive = mpeEvidNaive->GetValue(queryNode)->GetInt();

		// junction tree inference
		juncTreeInfEngine->EnterEvidence(evidence.get(), maximizeFlag);
		juncTreeInfEngine->MarginalNodes(&queryNode, 1);
		const pnl::CEvidence *mpeEvidJTree = juncTreeInfEngine->GetMPE();
		const int mpeValJTree = mpeEvidJTree->GetValue(queryNode)->GetInt();

		if (mpeValNaive != mpeValJTree)
		{
			// FIXME [implement] >>
		}
	}

	// start inference without maximization
	{
		const int maximizeFlag = 0;
		const int queryNode = 0;

		// naive inference
		naiveInfEngine->EnterEvidence(evidence.get(), maximizeFlag);
		naiveInfEngine->MarginalNodes(&queryNode, 1);
		const pnl::CPotential *marginalNaive = naiveInfEngine->GetQueryJPD();
		//marginalNaive->Dump();

		// junction tree inference
		juncTreeInfEngine->EnterEvidence(evidence.get(), maximizeFlag);
		juncTreeInfEngine->MarginalNodes(&queryNode, 1);
		const pnl::CPotential *marginalJTree = juncTreeInfEngine->GetQueryJPD();
		//marginalJTree->Dump();

		const float eps = 1.5e-1f;
		if (!marginalJTree->IsFactorsDistribFunEqual(marginalNaive, eps))
		{
			// TODO [implement] >>
			std::cout << "results of junction tree inference & naive inference are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;
		}
	}
}

pnl::CCPD * create_tabular_cpd_for_node_0(pnl::CModelDomain *md)
{
	// create tabular CPD for domain [0]
	const int numNodes = 1;
	const int domain[] = { 0 };
	const float table[] = { 0.7f, 0.3f };

	return pnl::CTabularCPD::Create(domain, numNodes, md, table);
}

pnl::CCPD * create_tabular_cpd_for_node_1(pnl::CModelDomain *md)
{
	// create tabular CPD for domain [1]
	const int numNodes = 1;
	const int domain[] = { 1 };
	const float table[] = { 0.4f, 0.6f };

	return pnl::CTabularCPD::Create(domain, numNodes, md, table);
}

pnl::CCPD * create_gaussian_cpd_for_node_2(pnl::CModelDomain *md)
{
	// create (univariate) Gaussian CPD for domain [2]
	const int domain[] = { 2 };
	const int numNodes = 1;
	const float mean0 = 0.0f;
	const float cov0 = 1.0f;

	pnl::CGaussianCPD *cpd = pnl::CGaussianCPD::Create(domain, numNodes, md);
	cpd->AllocDistribution(&mean0, &cov0, 1.0f, NULL);

	return cpd;
}

pnl::CCPD * create_mixture_of_gaussians_cpd_for_node_3(pnl::CModelDomain *md)
{
	// create mixture Gaussian CPD for domain [0, 1, 2, 3] -> univariate Gaussian
	// node 3 has the nodes 0, 1, 2 as parents
	// last discrete node among all discrete nodes in domain is the special node - mixture node
	// in this case node 1 is the mixture node
	const int numNodes = 4;
	const int domain[] = { 0, 1, 2, 3 };
	// this table must be equal to table for node 1, because node 1 (mixture node) is used for storage summing coefficients for Gaussians
	const float mixCoeffTable[] = { 0.4f, 0.6f };

	pnl::CMixtureGaussianCPD *cpd = pnl::CMixtureGaussianCPD::Create(domain, numNodes, md, mixCoeffTable);

	// data for probability distribution -> univariate Gaussian
	// discrete nodes 0 & 1 have 2 possible values.
	// if node 0 = 0 & node 1 = 0
	const float mean00 = 1.0f;
	const float cov00 = 0.005f;
	const float weight00 = 0.02f;

	// if node 0 = 1 & node 1 = 0
	const float mean10 = -5.0f;
	const float cov10 = 0.01f;
	const float weight10 = 0.01f;

	// if node 0 = 0 & node 1 = 1
	const float mean01 = -3.0f;
	const float cov01 = 0.01f;
	const float weight01 = 0.01f;

	// if node 0 = 1 & node 1 = 1
	const float mean11 = 2.0f;
	const float cov11 = 0.002f;
	const float weight11 = 0.05f;

	int parentVal[] = { 0, 0 };  // [ node 0 -> 0, node 1 -> 0 ]
	const float *dataWeight = &weight00;
	cpd->AllocDistribution(&mean00, &cov00, 2.0f, &dataWeight, parentVal);

	parentVal[1] = 1;  // [ node 0 -> 0, node 1 -> 1 ]
	dataWeight = &weight01;
	cpd->AllocDistribution(&mean01, &cov01, 1.0f, &dataWeight, parentVal);

	parentVal[0] = 1;  // [ node 0 -> 1, node 1 -> 1 ]
	dataWeight = &weight11;
	cpd->AllocDistribution(&mean11, &cov11, 1.0f, &dataWeight, parentVal);

	parentVal[1] = 0;  // [ node 0 -> 1, node 1 -> 0 ]
	dataWeight = &weight10;
	cpd->AllocDistribution(&mean10, &cov10, 1.0f, &dataWeight, parentVal);

	return cpd;
}

pnl::CCPD * create_gaussian_cpd_for_node_4(pnl::CModelDomain *md)
{
	// create (bivariate) Gaussian CPD for domain [3, 4]
	// node 4 has the node 3 as parent
	const int numNodes = 2;
	const int domain[] = { 3, 4 };

	// bivariate Gaussian
	const float mean[] = { 8.0f, 1.0f };
	const float cov[] = { 0.01f, 0.0f, 0.0f, 0.02f };
	const float weight[] = { 0.01f, 0.03f };

	pnl::CGaussianCPD *cpd = pnl::CGaussianCPD::Create(domain, numNodes, md);

	if (true)
	{
		const float *dataWeight = weight;
		cpd->AllocDistribution(mean, cov, 0.5f, &dataWeight);
	}
	else
	{
		// create factor using attach matrix

		int range[] = { 2, 1 };
		pnl::C2DNumericDenseMatrix<float> *meanMat = pnl::C2DNumericDenseMatrix<float>::Create(range, mean);
		cpd->AttachMatrix(meanMat, pnl::matMean);

		pnl::C2DNumericDenseMatrix<float> *weightMat = pnl::C2DNumericDenseMatrix<float>::Create(range, weight);
		cpd->AttachMatrix(weightMat, pnl::matWeights, 0);

		range[1] = 2;
		pnl::C2DNumericDenseMatrix<float> *covMat = pnl::C2DNumericDenseMatrix<float>::Create(range, cov);
		cpd->AttachMatrix(covMat, pnl::matCovariance);
	}

	return cpd;
}

// [ref] ${PNL_ROOT}/c_pgmtk/examples/mixture_gaussian_bnet/src/mixture_gaussian_bnet.cpp
pnl::CBNet * create_mixture_of_gaussians_bayesian_network()
{
/*
	let we want to create Bayesian network

	    A
	    |
	    v
	B-->C-->D

		where A is the discrete node; B & D are Gaussian; C is mixture Gaussian node

	In PNL graph for this model seems as

	    0  1
	    | /
	    v
	2-->3-->4

	where
		0 - tabular node corresponding to node A
		2 - Gaussian node corresponing to node B (univariate)
		4 - Gaussian node corresponing to node D (bivariate)

		1 - tabular nodes. It is a special node - mixture node.
			it is used for storage summing coefficients for Gaussians.
			it must be a last discrete node among all discrete parents for Gaussian mixture node.
		3 - Gaussain mixture node. (univariate)
		nodes 1 & 3 together correspond to node C.
*/

	// 1) first need to specify the graph structure of the model;

	const int numNodes = 5;
	const int numNeighbors[numNodes] = { 1, 1, 1, 4, 1 };

	const int nbrs0[] = { 3 };
	const int nbrs1[] = { 3 };
	const int nbrs2[] = { 3 };
	const int nbrs3[] = { 0, 1, 2, 4 };
	const int nbrs4[] = { 3 };
	const int *nbrs[] = { nbrs0, nbrs1, nbrs2, nbrs3, nbrs4 };

	// neighbors can be of either one of three following types:
	// a parent, a child or just a neighbor - for undirected graphs.
	// if a neighbor of a node is it's parent, then neighbor type is pnl::ntParent
	// if it's a child, then pnl::ntChild and if it's a neighbor, then pnl::ntNeighbor
	const pnl::ENeighborType nbrsTypes0[] = { pnl::ntChild };
	const pnl::ENeighborType nbrsTypes1[] = { pnl::ntChild };
	const pnl::ENeighborType nbrsTypes2[] = { pnl::ntChild };
	const pnl::ENeighborType nbrsTypes3[] = { pnl::ntParent, pnl::ntParent, pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType nbrsTypes4[] = { pnl::ntParent };
	const pnl::ENeighborType *nbrsTypes[] = { nbrsTypes0, nbrsTypes1, nbrsTypes2, nbrsTypes3, nbrsTypes4 };

	pnl::CGraph *graph = pnl::CGraph::Create(numNodes, numNeighbors, nbrs, nbrsTypes);

	// 2) creation of the model domain.
	//	there are 3 several types of nodes in the example:
	//	1) discrete nodes 0 & 1 with 2 possible values
	//	2) scalar continuous (Gaussian) nodes 2, 3
	//	3) multivariate Gaussian node 4 (consists of 2 values)
	const int numNodeTypes = 3;
	pnl::nodeTypeVector nodeTypes(numNodeTypes);
	nodeTypes[0].SetType(true, 2);  // discrete & binary
	nodeTypes[1].SetType(false, 1);  // continuous & univariate
	nodeTypes[2].SetType(false, 2);  // continuous & bivariate

	const int nnodes = graph->GetNumberOfNodes();
	pnl::intVector nodeAssociation(nnodes, 1);  // { 0, 0, 1, 1, 2 }
	nodeAssociation[0] = 0;
	nodeAssociation[1] = 0;
	nodeAssociation[4] = 2;

	pnl::CModelDomain *md = pnl::CModelDomain::Create(nodeTypes, nodeAssociation);

	// 2) creation base for BNet using Graph, and Model Domain
	pnl::CBNet *bnet = pnl::CBNet::Create(graph, md);

	// 3) allocation space for all factors of the model
	bnet->AllocFactors();

	// 4) creation factors and attach their to model
	pnl::CCPD *cpd = create_tabular_cpd_for_node_0(md);
	bnet->AttachFactor(cpd);

	cpd = create_tabular_cpd_for_node_1(md);
	bnet->AttachFactor(cpd);

	cpd = create_gaussian_cpd_for_node_2(md);
	bnet->AttachFactor(cpd);

	cpd = create_mixture_of_gaussians_cpd_for_node_3(md);
	bnet->AttachFactor(cpd);

	cpd = create_gaussian_cpd_for_node_4(md);
	bnet->AttachFactor(cpd);

	return bnet;
}

void learn_mixture_of_gaussians_bayesian_network(const boost::scoped_ptr<pnl::CBNet> &mogBNet)
{
	// FIXME [implement] >>
	throw std::runtime_error("not yet implemented");
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/AMixtureGaussainModel.cpp
void infer_mixture_of_gaussians_bayesian_network_1(const boost::scoped_ptr<pnl::CBNet> &mogBNet)
{
	// create evidence on all Gaussian nodes

	//const int numNodes = mogBNet->GetNumberOfNodes();

	// create evidence for inference
	const int numObsNodes = 3;
	const int obsNodes[] = { 2, 3, 4 };

	pnl::valueVector obsVals(numObsNodes, (pnl::Value)0);
	obsVals[0].SetFlt(-0.5f);
	obsVals[1].SetFlt(2.0f);
	obsVals[2].SetFlt(1.0f);

	boost::scoped_ptr<pnl::CEvidence> evidence(pnl::CEvidence::Create(mogBNet.get(), numObsNodes, obsNodes, obsVals));

	// create inference engine
	boost::scoped_ptr<pnl::CNaiveInfEngine> naiveInfEngine(pnl::CNaiveInfEngine::Create(mogBNet.get()));
	boost::scoped_ptr<pnl::CJtreeInfEngine> juncTreeInfEngine(pnl::CJtreeInfEngine::Create(mogBNet.get()));

	// start inference with maximization
	{
		const int maximizeFlag = 1;
		const int queryNode = 0;
		const int queryNodes[] = { 0, 1 };

		// naive inference
		naiveInfEngine->EnterEvidence(evidence.get(), maximizeFlag);
#if 1
		naiveInfEngine->MarginalNodes(&queryNode, 1);
		const pnl::CEvidence *mpeEvidNaive = naiveInfEngine->GetMPE();
		const int mpeValNaive = mpeEvidNaive->GetValue(queryNode)->GetInt();
#else
		naiveInfEngine->MarginalNodes(queryNodes, 2);
		const pnl::CEvidence *mpeEvidNaive = naiveInfEngine->GetMPE();
		const int mpeValNaive0 = mpeEvidNaive->GetValue(queryNodes[0])->GetInt();
		const int mpeValNaive1 = mpeEvidNaive->GetValue(queryNodes[1])->GetInt();
#endif

		// junction tree inference
		juncTreeInfEngine->EnterEvidence(evidence.get(), maximizeFlag);
#if 1
		juncTreeInfEngine->MarginalNodes(&queryNode, 1);
		const pnl::CEvidence *mpeEvidJTree = juncTreeInfEngine->GetMPE();
		const int mpeValJTree = mpeEvidJTree->GetValue(queryNode)->GetInt();

		if (mpeValNaive != mpeValJTree)
		{
			// FIXME [implement] >>
		}
#else
		juncTreeInfEngine->MarginalNodes(queryNodes, 2);
		const pnl::CEvidence *mpeEvidJTree = juncTreeInfEngine->GetMPE();
		const int mpeValJTree0 = mpeEvidJTree->GetValue(queryNodes[0])->GetInt();
		const int mpeValJTree1 = mpeEvidJTree->GetValue(queryNodes[1])->GetInt();
#endif
	}

	// start inference without maximization
	{
		const int maximizeFlag = 0;
		const int queryNode = 0;
		const int queryNodes[] = { 0, 1 };

		// naive inference
		naiveInfEngine->EnterEvidence(evidence.get(), maximizeFlag);
#if 1
		naiveInfEngine->MarginalNodes(&queryNode, 1);
		const pnl::CPotential *marginalNaive = naiveInfEngine->GetQueryJPD();
#else
		naiveInfEngine->MarginalNodes(queryNodes, 2);
		const pnl::CPotential *marginalNaive = naiveInfEngine->GetQueryJPD();
#endif
		//marginalNaive->Dump();

		// junction tree inference
		juncTreeInfEngine->EnterEvidence(evidence.get(), maximizeFlag);
#if 1
		juncTreeInfEngine->MarginalNodes(&queryNode, 1);
		const pnl::CPotential *marginalJTree = juncTreeInfEngine->GetQueryJPD();
#else
		juncTreeInfEngine->MarginalNodes(queryNodes, 2);
		const pnl::CPotential *marginalJTree = juncTreeInfEngine->GetQueryJPD();
#endif
		//marginalJTree->Dump();

		const float eps = 1.5e-1f;
		if (!marginalJTree->IsFactorsDistribFunEqual(marginalNaive, eps))
		{
			// TODO [implement] >>
			std::cout << "results of junction tree inference & naive inference are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;
		}
	}
}

void infer_mixture_of_gaussians_bayesian_network_2(const boost::scoped_ptr<pnl::CBNet> &mogBNet)
{
	// create simple evidence for nodes 2, 3, 4 from BNet
	const pnl::CModelDomain *modelDomain = mogBNet->GetModelDomain();

	// let nodes 2, 3, & 4 be observed
	const int numObsNodes = 3;
	const int obsNodes[] = { 2, 3, 4 };

	//
	int numObsValues = 0;
	for (int i = 0; i < numObsNodes; ++i)
	{
		numObsValues += modelDomain->GetNumVlsForNode(obsNodes[i]);
	}
	// evidence for node 2 consists of 1 value
	// evidence for node 3 consists of 1 value
	// evidence for node 4 consists of 2 values
	// so, numObsValues = 4

	pnl::valueVector obsValues(numObsValues);
	for (int i = 0; i < numObsValues; ++i)
	{
		const float val = pnl::pnlRand(-1.0f, 1.0f);
		obsValues[i].SetFlt(val);
	}

	boost::scoped_ptr<pnl::CEvidence> evidence(pnl::CEvidence::Create(modelDomain, numObsNodes, obsNodes, obsValues));

	// create junction tree inference engine
	boost::scoped_ptr<pnl::CJtreeInfEngine> juncTreeInfEngine(pnl::CJtreeInfEngine::Create(mogBNet.get()));

	// enter evidence created before and started inference procedure
	juncTreeInfEngine->EnterEvidence(evidence.get());

	// get a marginal for query
	const int numQueryNodes = 1;
	const int queryNodes[] = { 0 };
	juncTreeInfEngine->MarginalNodes(queryNodes, numQueryNodes);
	const pnl::CPotential *queryPot = juncTreeInfEngine->GetQueryJPD();
	// node 0 is discrete, then query potential is tabular

	const pnl::CDistribFun *distribFun = queryPot->GetDistribFun();
	const pnl::CMatrix<float> *queryPotMat = distribFun->GetMatrix(pnl::matTable);

	const int node0Size = modelDomain->GetVariableType(0)->GetNodeSize();

	for (int index = 0; index < node0Size; ++index)
	{
		const float val = queryPotMat->GetElementByIndexes(&index);
		std::cout << " Probability of event node 0 take on a value ";
		std::cout << index << " is ";
		std::cout << val << std::endl;
	}

	// distribution of the query hase dense matrix
	// the row data is
	int numElem;
	const float *data;
	static_cast<const pnl::CNumericDenseMatrix<float> *>(queryPotMat)->GetRawData(&numElem, &data);
	std::cout << " The raw data of the query distribution is: ";

	for (int i = 0; i < numElem; ++i)
	{
		std::cout << data[i] << " ";
	}
	std::cout << std::endl;

	const float loglik = juncTreeInfEngine->GetLogLik();
	std::cout << " Log Likelihood of the evidence is " << loglik << std::endl;
}

}  // namespace local
}  // unnamed namespace

void bayesian_network()
{
	// discrete Bayesian network
	std::cout << "========== discrete Bayesian network" << std::endl;
	{
		const boost::scoped_ptr<pnl::CBNet> discreteBNet(local::create_discrete_bayesian_network());

		if (!discreteBNet)
		{
			std::cout << "fail to create a probabilistic graphical model at " << __LINE__ << " in " << __FILE__ << std::endl;
			return;
		}

		//boost::scoped_ptr<pnl::CJtreeInfEngine> juncTreeInfEngine(pnl::CJtreeInfEngine::Create(discreteBNet.get()));  // runtime-error
	}

	std::cout << "\n========== single mixture-of-Gaussians Bayesian network" << std::endl;
	// single mixture-of-Gaussians Bayesian network
	{
		// create single mixture-of-Gaussians BNet
		//const boost::scoped_ptr<pnl::CBNet> mogBNet(local::create_single_mixture_of_gaussians_bayesian_network());
		const boost::scoped_ptr<pnl::CBNet> mogBNet(pnl::pnlExCreateSingleGauMix());

		if (!mogBNet)
		{
			std::cout << "fail to create a probabilistic graphical model at " << __LINE__ << " in " << __FILE__ << std::endl;
			return;
		}

		local::learn_single_mixture_of_gaussians_bayesian_network(mogBNet);  // run-time error: to be verified
		local::infer_single_mixture_of_gaussians_bayesian_network(mogBNet);  // run-time error: to be verified
	}

	std::cout << "\n========== mixture-of-Gaussians Bayesian network" << std::endl;
	// mixture-of-Gaussians Bayesian network
	{
		// create mixture-of-Gaussians BNet
		const boost::scoped_ptr<pnl::CBNet> mogBNet(local::create_mixture_of_gaussians_bayesian_network());
		//const boost::scoped_ptr<pnl::CBNet> discreteBNet(pnl::pnlExCreateVerySimpleGauMix());
		//const boost::scoped_ptr<pnl::CBNet> discreteBNet(pnl::pnlExCreateSimpleGauMix());

		if (!mogBNet)
		{
			std::cout << "fail to create a probabilistic graphical model at " << __LINE__ << " in " << __FILE__ << std::endl;
			return;
		}

		{
			std::cout << " graph of mixture-of-Gaussians Bayesian network";
			mogBNet->GetGraph()->Dump();
		}

		//local::learn_mixture_of_gaussians_bayesian_network(mogBNet);  // not yet implemented
#if 0
		local::infer_mixture_of_gaussians_bayesian_network_1(mogBNet);
#else
		local::infer_mixture_of_gaussians_bayesian_network_2(mogBNet);
#endif
	}
}
