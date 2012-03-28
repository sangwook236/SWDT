//#include "stdafx.h"
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

	where
		X0, X1 - tabular nodes (bivariate)
		Y0, Y1 - tabular nodes (trivariate)
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
	const int numNeighs[] = {
		2, 1,  // 1st time-slice
		2, 1  // 2nd time-slice
	};

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
	nodeTypes[1].SetType(true, 3);  // discrete & trinary

	pnl::intVector nodeAssociation(numNodes);
	nodeAssociation[0] = 0;
	nodeAssociation[1] = 1;
	nodeAssociation[2] = 0;
	nodeAssociation[3] = 1;

	pnl::CBNet *bnet = pnl::CBNet::Create(numNodes, nodeTypes, nodeAssociation, graph);

	// create raw data tables for CPDs
	const float table0[] = { 0.6f, 0.4f };
	const float table1[] = { 0.1f, 0.4f, 0.5f, 0.6f, 0.3f, 0.1f };
	const float table2[] = { 0.7f, 0.3f, 0.4f, 0.6f };
	const float table3[] = { 0.1f, 0.4f, 0.5f, 0.6f, 0.3f, 0.1f };

	// create factors and attach their to model
	bnet->AllocFactors();
	bnet->AllocFactor(0);
	bnet->GetFactor(0)->AllocMatrix(table0, pnl::matTable);
	bnet->AllocFactor(1);
	bnet->GetFactor(1)->AllocMatrix(table1, pnl::matTable);
	bnet->AllocFactor(2);
	bnet->GetFactor(2)->AllocMatrix(table2, pnl::matTable);
	bnet->AllocFactor(3);
	bnet->GetFactor(3)->AllocMatrix(table3, pnl::matTable);
#else
	pnl::CBNet *bnet = pnl::pnlExCreateRndArHMM();
#endif

	// create DBN
	return pnl::CDBN::Create(bnet);
}

// [ref] pnlExCreateCondGaussArBNet() & pnlExCreateRndArHMM() in ${PNL_ROOT}/c_pgmtk/src/pnlExampleModels.cpp
pnl::CDBN * create_hmm_with_ar_gaussian_observations()
{
/*
	an HMM with autoregressive Gaussian observations

		X0 -> X1
		|     |
		v     v
		Y0 -> Y1

	where
		X0, X1 - tabular nodes (bivariate)
		Y0, Y1 - Gaussian nodes (univariate)
*/

	// create static model
	const int numNodes = 4;  // number of nodes
	const int numNodeTypes = 2;  // number of node types (all nodes are discrete)

	// create a DAG
	const int numNeighs[] = {
		2, 2,  // 1st time-slice
		2, 2  // 2nd time-slice
	};

	const int neigh0[] = { 1, 2 };
	const int neigh1[] = { 0, 3 };
	const int neigh2[] = { 0, 3 };
	const int neigh3[] = { 1, 2 };
	const int *neighs[] = { neigh0, neigh1, neigh2, neigh3 };

	const pnl::ENeighborType orient0[] = { pnl::ntChild, pnl::ntChild };
	const pnl::ENeighborType orient1[] = { pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient2[] = { pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient3[] = { pnl::ntParent, pnl::ntParent };
	const pnl::ENeighborType *orients[] = { orient0, orient1, orient2, orient3 };

	pnl::CGraph *graph = pnl::CGraph::Create(numNodes, numNeighs, neighs, orients);

	// create static BNet
/*
	pnl::CNodeType *nodeTypes = new pnl::CNodeType [numNodeTypes];
	nodeTypes[0].SetType(true, 2);  // discrete & binary
	nodeTypes[1].SetType(false, 1);  // continuous & univariate

	const int nodeAssociation[] = { 0, 1, 0, 1 };

	pnl::CBNet *bnet = pnl::CBNet::Create(numNodes, numNodeTypes, nodeTypes, nodeAssociation, graph);

	delete [] nodeTypes;
	nodeTypes = NULL;
*/
	pnl::nodeTypeVector nodeTypes(numNodeTypes);
	nodeTypes[0].SetType(true, 2);  // discrete & binary
	nodeTypes[1].SetType(false, 1);  // continuous & univariate

	pnl::intVector nodeAssociation(numNodes);
	nodeAssociation[0] = 0;
	nodeAssociation[1] = 1;
	nodeAssociation[2] = 0;
	nodeAssociation[3] = 1;

	pnl::CBNet *bnet = pnl::CBNet::Create(numNodes, nodeTypes, nodeAssociation, graph);

	//
	bnet->AllocFactors();

	// let arbitrary distribution is
#if 1
	const float tableNode0[] = { 0.95f, 0.05f };
	const float tableNode2[] = { 0.1f, 0.9f, 0.5f, 0.5f };
#else
	const float tableNode0[] = { 0.7f, 0.3f };
    const float tableNode2[] = { 0.2f, 0.8f, 0.3f, 0.7f };
#endif

	const float mean1w0 = -3.2f;  const float cov1w0 = 0.00002f;
	const float mean1w1 = -0.5f;  const float cov1w1 = 0.0001f;

	const float mean3w0 = 6.5f;  const float cov3w0 = 0.03f;  const float weight3w0 = 1.0f;
	const float mean3w1 = 7.5f;  const float cov3w1 = 0.04f;  const float weight3w1 = 0.5f;

	bnet->AllocFactor(0);
	bnet->GetFactor(0)->AllocMatrix(tableNode0, pnl::matTable);

	bnet->AllocFactor(1);
	int parentVal[] = { 0 };
	bnet->GetFactor(1)->AllocMatrix(&mean1w0, pnl::matMean, -1, parentVal);
	bnet->GetFactor(1)->AllocMatrix(&cov1w0, pnl::matCovariance, -1, parentVal);
	parentVal[0] = 1;
	bnet->GetFactor(1)->AllocMatrix(&mean1w1, pnl::matMean, -1, parentVal);
	bnet->GetFactor(1)->AllocMatrix(&cov1w1, pnl::matCovariance, -1, parentVal);

	bnet->AllocFactor(2);
	bnet->GetFactor(2)->AllocMatrix(tableNode2, pnl::matTable);

	bnet->AllocFactor(3);
	parentVal[0] = 0;
	bnet->GetFactor(3)->AllocMatrix(&mean3w0, pnl::matMean, -1, parentVal);
	bnet->GetFactor(3)->AllocMatrix(&cov3w0, pnl::matCovariance, -1, parentVal);
	bnet->GetFactor(3)->AllocMatrix(&weight3w0, pnl::matWeights, 0, parentVal);
	parentVal[0] = 1;
	bnet->GetFactor(3)->AllocMatrix(&mean3w1, pnl::matMean, -1, parentVal);
	bnet->GetFactor(3)->AllocMatrix(&cov3w1, pnl::matCovariance, -1, parentVal);
	bnet->GetFactor(3)->AllocMatrix(&weight3w1, pnl::matWeights, 0, parentVal);

	// create DBN using BNet
	return pnl::CDBN::Create(bnet);
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
	pnl::pEvidencesVector evidencesForDBN(numTimeSlices);
	for (int time_slice = 0; time_slice < numTimeSlices; ++time_slice)
	{
		obsNodesVals[0].SetInt(observations[time_slice]);
		evidencesForDBN[time_slice] = pnl::CEvidence::Create(hmm.get(), obsNodes, obsNodesVals);
	}

	// create an inference engine
	boost::scoped_ptr<pnl::C1_5SliceJtreeInfEngine> infEngine(pnl::C1_5SliceJtreeInfEngine::Create(hmm.get()));

	// create inference (smoothing) for DBN
	infEngine->DefineProcedure(pnl::ptViterbi, numTimeSlices);
	infEngine->EnterEvidence(&evidencesForDBN.front(), numTimeSlices);
	infEngine->FindMPE();

	pnl::intVector queryPrior(1), query(2);
	queryPrior[0] = 0;  // 0th node ==> hidden state
	query[0] = 0;  query[1] = 2;  // 0th & 2nd nodes ==> hidden states

	for (int time_slice = 0; time_slice < numTimeSlices; ++time_slice)
	{
		if (time_slice)  // for the transition network
		{
			infEngine->MarginalNodes(&query.front(), query.size(), time_slice);
		}
		else  // for the prior network
		{
			infEngine->MarginalNodes(&queryPrior.front(), queryPrior.size(), time_slice);
		}

		const pnl::CPotential *queryMPE = infEngine->GetQueryMPE();

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
		const pnl::CEvidence *mpeEvid = infEngine->GetMPE();
		std::cout << " MPE node value: ";
#if 0
		for (int i = 0; i < numNodes; ++i)
		{
			const int mpeNodeVal = mpeEvid->GetValue(domain[i])->GetInt();
			std::cout << mpeNodeVal << " ";
		}
		std::cout << std::endl;
#else
		const int mpeNodeVal = mpeEvid->GetValue(domain[numNodes-1])->GetInt();
		std::cout << mpeNodeVal << std::endl;
#endif
	}

	//
	for (int i = 0; i < numTimeSlices; ++i)
	{
		delete evidencesForDBN[i];
	}
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/ALearningCondGaussDBN.cpp
void learn_hmm_with_ar_gaussian_observations(const boost::scoped_ptr<pnl::CDBN> &dbn)
{
    //pnl::CBNet *bnetDBN = pnl::pnlExCreateCondGaussArBNet();
	//const boost::scoped_ptr<pnl::CDBN> dbn(pnl::CDBN::Create(bnetDBN));
	const pnl::CBNet *bnetDBN = dynamic_cast<const pnl::CBNet *>(dbn->GetStaticModel());

	//
	pnl::CGraph *graphDBN = pnl::CGraph::Copy(bnetDBN->GetGraph());
	pnl::CModelDomain *modelDomainDBN = bnetDBN->GetModelDomain();

	const boost::scoped_ptr<pnl::CBNet> bnetToLearn(pnl::CBNet::CreateWithRandomMatrices(graphDBN, modelDomainDBN));
	const boost::scoped_ptr<pnl::CDBN> dbnToLearn(pnl::CDBN::Create(pnl::CBNet::Copy(bnetToLearn.get())));

	// the definitions of time series & time slices
	//	[ref] "Probabilistic Network Library: User Guide and Reference Manual", pp. 3-290 & 2-32

	// generate sample
	const int numTimeSeries = 5000;

	const pnl::intVector numSlices(numTimeSeries, 2);
	pnl::pEvidencesVecVector evidencesForDBN;
	dbn->GenerateSamples(&evidencesForDBN, numSlices);

	pnl::pEvidencesVector evidencesForBNet;
	bnetToLearn->GenerateSamples(&evidencesForBNet, numTimeSeries);

	for (int i = 0; i < numTimeSeries; ++i)
	{
		pnl::valueVector vls1, vls2;
		(evidencesForDBN[i])[0]->GetRawData(&vls1);
		(evidencesForDBN[i])[1]->GetRawData(&vls2);

		pnl::valueVector newData(vls1.size() + vls2.size());
		std::memcpy(&newData.front(), &vls1.front(), vls1.size() * sizeof(pnl::Value));
		std::memcpy(&newData[vls1.size()], &vls2.front(), vls2.size() * sizeof(pnl::Value));

		evidencesForBNet[i]->SetData(newData);

		//(evidencesForDBN[i])[0]->MakeNodeHiddenBySerialNum(0);
		//(evidencesForDBN[i])[1]->MakeNodeHiddenBySerialNum(0);

		//evidencesForBNet[i]->MakeNodeHiddenBySerialNum(0);
		//evidencesForBNet[i]->MakeNodeHiddenBySerialNum(2);
	}

	// forbids matrix change in learning process
	const int parentIndices[] = { 0 };
	bnetToLearn->GetFactor(1)->GetDistribFun()->GetMatrix(pnl::matMean, -1, parentIndices)->SetClamp(1);
	bnetToLearn->GetFactor(3)->GetDistribFun()->GetMatrix(pnl::matCovariance, -1, parentIndices)->SetClamp(1);
	bnetToLearn->GetFactor(3)->GetDistribFun()->GetMatrix(pnl::matCovariance, 0, parentIndices)->SetClamp(1);

	// learn
	const int maxIteration = 10;

	const boost::scoped_ptr<pnl::CEMLearningEngine> learnerForBNet(pnl::CEMLearningEngine::Create(bnetToLearn.get()));
	learnerForBNet->SetData(numTimeSeries, &evidencesForBNet.front());
	learnerForBNet->SetMaxIterEM(maxIteration);
	try
	{
		learnerForBNet->Learn();
	}
	catch (const pnl::CAlgorithmicException &e)
	{
		std::cout << "fail to learn parameters of a unrolled Bayesian network of a HMM w/ AR Gaussian observations" << e.GetMessage() << std::endl;
		return;
	}

	const boost::scoped_ptr<pnl::CEMLearningEngineDBN> learnerForDBN(pnl::CEMLearningEngineDBN::Create(dbnToLearn.get()));
	learnerForDBN->SetData(evidencesForDBN);
	learnerForDBN->SetMaxIterEM(maxIteration);
	try
	{
		learnerForDBN->Learn();
	}
	catch (const pnl::CAlgorithmicException &e)
	{
		std::cout << "fail to learn parameters of a HMM w/ AR Gaussian observations" << e.GetMessage() << std::endl;
		return;
	}

	//
	const pnl::CMatrix<float> *matBNet = bnetToLearn->GetFactor(0)->GetDistribFun()->GetStatisticalMatrix(pnl::stMatTable);
	const pnl::CMatrix<float> *matDBN = dbnToLearn->GetFactor(0)->GetDistribFun()->GetStatisticalMatrix(pnl::stMatTable);

	const float eps = 0.1f;
	for (int i = 0; i < 4; ++i)
	{
		std::cout << "\n ___ node " << i << "_________________________" << std::endl;
		std::cout << "\n____ BNet_________________________________" << std::endl;
		bnetToLearn->GetFactor(i)->GetDistribFun()->Dump();

		if (!bnetDBN->GetFactor(i)->IsFactorsDistribFunEqual(dbnToLearn->GetFactor(i), eps, 0))
		{
			std::cout << "original & learned models are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;

			std::cout << "\n____ DBN__________________________________" << std::endl;
			dbnToLearn->GetFactor(i)->GetDistribFun()->Dump();
			std::cout << "\n____ initial DBN__________________________" << std::endl;
			bnetDBN->GetFactor(i)->GetDistribFun()->Dump();
			std::cout << "\n___________________________________________" << std::endl;
		}
	}

	//
	for (int i = 0; i < numTimeSeries; ++i)
	{
		for (size_t j = 0; j < evidencesForDBN[i].size(); ++j)
			delete (evidencesForDBN[i])[j];

		delete evidencesForBNet[i];
	}
}

// [ref] create_dbn_with_mixture_of_gaussians_observations() in dbn.cpp
pnl::CDBN * create_hmm_with_mixture_of_gaussians_observations()
{
/*
	an HMM with mixture-of-Gaussians observations

		    0 ---> 3
		    |      |
		    |      |
		    v      v
	    1-->2  4-->5

	where
		0, 3 - tabular nodes (k-variate)
		2, 5 - Gaussian mixture nodes (univariate)
		1, 4 - tabular nodes. they are special nodes - mixture nodes (p-variate)
			   they are used for storage summing coefficients for Gaussians.
			   they must be last discrete nodes among all discrete parents for Gaussian mixture node.
*/

	// create static model
	const int numNodes = 6;
	const int numNodeTypes = 3;

	// TODO [check] >> these are magic numbers
	const int numStates = 2;  // k-variate
	const int numMixtures = 2;  // p-variate

	// create a DAG
	const int numNeighs[] = {
		2, 1, 2,  // 1st time-slice
		2, 1, 2  // 2nd time-slice
	};

	const int neigh0[] = { 2, 3 };
	const int neigh1[] = { 2 };
	const int neigh2[] = { 0, 1 };
	const int neigh3[] = { 0, 5 };
	const int neigh4[] = { 5 };
	const int neigh5[] = { 3, 4 };
	const int *neighs[] = { neigh0, neigh1, neigh2, neigh3, neigh4, neigh5 };

	const pnl::ENeighborType orient0[] = { pnl::ntChild, pnl::ntChild };
	const pnl::ENeighborType orient1[] = { pnl::ntChild };
	const pnl::ENeighborType orient2[] = { pnl::ntParent, pnl::ntParent };
	const pnl::ENeighborType orient3[] = { pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType orient4[] = { pnl::ntChild };
	const pnl::ENeighborType orient5[] = { pnl::ntParent, pnl::ntParent };
	const pnl::ENeighborType *orients[] = { orient0, orient1, orient2, orient3, orient4, orient5 };

	pnl::CGraph *graph = pnl::CGraph::Create(numNodes, numNeighs, neighs, orients);

	// create static BNet
	pnl::nodeTypeVector nodeTypes(numNodeTypes);
	nodeTypes[0].SetType(true, numStates);  // discrete & k-nary
	nodeTypes[1].SetType(true, numMixtures);  // discrete & p-nary
	nodeTypes[2].SetType(false, 1);  // continous & univariate

	pnl::intVector nodeAssociation(numNodes);
	nodeAssociation[0] = 0;
	nodeAssociation[1] = 1;
	nodeAssociation[2] = 2;
	nodeAssociation[3] = 0;
	nodeAssociation[4] = 1;
	nodeAssociation[5] = 2;

	//
#if 1
	pnl::CModelDomain *modelDomain = pnl::CModelDomain::Create(nodeTypes, nodeAssociation);

	// to be learned
	pnl::CBNet *bnet = pnl::CBNet::CreateWithRandomMatrices(graph, modelDomain);
#else
    // FIXME [correct] >> compile-time error
    //  I have no idea of this cause

	pnl::CBNet *bnet = pnl::CBNet::Create(numNodes, nodeTypes, nodeAssociation, graph);

	//
	bnet->AllocFactors();

	// FIXME [check] >> these parameters are to be determined
	const float tableNode0[] = { 0.95f, 0.05f };
	const float tableNode1[] = { 0.95f, 0.05f };
	const float tableNode3[] = { 0.1f, 0.9f, 0.1f, 0.9f };
	const float tableNode4[] = { 0.1f, 0.9f };

	const float mean2w00 = -3.2f, cov2w00 = 0.00002f;  // node2 for node0 = 0 & node1 = 0
	const float mean2w10 = -0.5f, cov2w10 = 0.0001f;  // node2 for node0 = 1 & node1 = 0
	const float mean2w01 = -3.2f, cov2w01 = 0.00002f;  // node2 for node0 = 0 & node1 = 1
	const float mean2w11 = -0.5f, cov2w11 = 0.0001f;  // node2 for node0 = 1 & node1 = 1

	const float mean5w00 = 6.5f, cov5w00 = 0.03f, weight5w00 = 1.0f;  // node5 for node3 = 0 & node4 = 0
	const float mean5w10 = 7.5f, cov5w10 = 0.04f, weight5w10 = 0.5f;  // node5 for node3 = 1 & node4 = 0
	const float mean5w01 = 6.5f, cov5w01 = 0.03f, weight5w01 = 1.0f;  // node5 for node3 = 0 & node4 = 1
	const float mean5w11 = 7.5f, cov5w11 = 0.04f, weight5w11 = 0.5f;  // node5 for node3 = 1 & node4 = 1

#if 0
    // FIXME [correct] >> run-time error
    //  I have no idea of this cause

	bnet->AllocFactor(0);
	bnet->GetFactor(0)->AllocMatrix(tableNode0, pnl::matTable);

	bnet->AllocFactor(1);
	bnet->GetFactor(1)->AllocMatrix(tableNode1, pnl::matTable);

	bnet->AllocFactor(2);
	int parentVal[] = { 0, 0 };  // node0 = 0 & node1 = 0
	bnet->GetFactor(2)->AllocMatrix(&mean2w00, pnl::matMean, -1, parentVal);
	bnet->GetFactor(2)->AllocMatrix(&cov2w00, pnl::matCovariance, -1, parentVal);
	//bnet->GetFactor(2)->AllocMatrix(&weight2w00, pnl::matWeights, 0, parentVal);
	parentVal[1] = 1;  // node0 = 0 & node1 = 1
	bnet->GetFactor(2)->AllocMatrix(&mean2w01, pnl::matMean, -1, parentVal);
	bnet->GetFactor(2)->AllocMatrix(&cov2w01, pnl::matCovariance, -1, parentVal);
	//bnet->GetFactor(2)->AllocMatrix(&weight2w01, pnl::matWeights, 0, parentVal);
	parentVal[0] = 1;  // node0 = 1 & node1 = 1
	bnet->GetFactor(2)->AllocMatrix(&mean2w11, pnl::matMean, -1, parentVal);
	bnet->GetFactor(2)->AllocMatrix(&cov2w11, pnl::matCovariance, -1, parentVal);
	//bnet->GetFactor(2)->AllocMatrix(&weight2w11, pnl::matWeights, 0, parentVal);
	parentVal[1] = 0;  // node0 = 1 & node1 = 0
	bnet->GetFactor(2)->AllocMatrix(&mean2w10, pnl::matMean, -1, parentVal);
	bnet->GetFactor(2)->AllocMatrix(&cov2w10, pnl::matCovariance, -1, parentVal);
	//bnet->GetFactor(2)->AllocMatrix(&weight2w10, pnl::matWeights, 0, parentVal);

	bnet->AllocFactor(3);
	bnet->GetFactor(3)->AllocMatrix(tableNode3, pnl::matTable);

	bnet->AllocFactor(4);
	bnet->GetFactor(4)->AllocMatrix(tableNode4, pnl::matTable);

	bnet->AllocFactor(5);
	parentVal[0] = parentVal[1] = 0;  // node3 = 0 & node4 = 0
	bnet->GetFactor(5)->AllocMatrix(&mean5w00, pnl::matMean, -1, parentVal);
	bnet->GetFactor(5)->AllocMatrix(&cov5w00, pnl::matCovariance, -1, parentVal);
    std::cout << "***** 111" << std::endl;  // FIXME [delete] >>
	bnet->GetFactor(5)->AllocMatrix(&weight5w00, pnl::matWeights, -1, parentVal);
    std::cout << "***** 222" << std::endl;  // FIXME [delete] >>
	parentVal[1] = 1;  // node3 = 0 & node4 = 1
	bnet->GetFactor(5)->AllocMatrix(&mean5w01, pnl::matMean, -1, parentVal);
	bnet->GetFactor(5)->AllocMatrix(&cov5w01, pnl::matCovariance, -1, parentVal);
	bnet->GetFactor(5)->AllocMatrix(&weight5w01, pnl::matWeights, -1, parentVal);
	parentVal[0] = 1;  // node3 = 1 & node4 = 1
	bnet->GetFactor(5)->AllocMatrix(&mean5w11, pnl::matMean, -1, parentVal);
	bnet->GetFactor(5)->AllocMatrix(&cov5w11, pnl::matCovariance, -1, parentVal);
	bnet->GetFactor(5)->AllocMatrix(&weight5w11, pnl::matWeights, -1, parentVal);
	parentVal[1] = 0;  // node3 = 1 & node4 = 0
	bnet->GetFactor(5)->AllocMatrix(&mean5w10, pnl::matMean, -1, parentVal);
	bnet->GetFactor(5)->AllocMatrix(&cov5w10, pnl::matCovariance, -1, parentVal);
	bnet->GetFactor(5)->AllocMatrix(&weight5w10, pnl::matWeights, -1, parentVal);
    std::cout << "***** 333" << std::endl;  // FIXME [delete] >>
#else
	const int domain0[] = { 0 };
	const int domain1[] = { 1 };
	const int domain2[] = { 0, 1, 2 };
	const int domain3[] = { 0, 3 };
	const int domain4[] = { 4 };
	const int domain5[] = { 3, 4, 5 };
	const int *domains[] = { domain0, domain1, domain2, domain3, domain4, domain5 };

	std::vector<int> numDomains(numNodes, 0);
	numDomains[0] = 1;
	numDomains[1] = 1;
	numDomains[2] = 3;
	numDomains[3] = 2;
	numDomains[4] = 1;
	numDomains[5] = 3;

	pnl::CTabularCPD *cpd0 = pnl::CTabularCPD::Create(domains[0], numDomains[0], bnet->GetModelDomain(), tableNode0);
	bnet->AttachFactor(cpd0);

	pnl::CTabularCPD *cpd1 = pnl::CTabularCPD::Create(domains[1], numDomains[1], bnet->GetModelDomain(), tableNode1);
	bnet->AttachFactor(cpd1);

    // FIXME [check] >> what does the 'normCoeff' parameter in CMixtureGaussianCPD::AllocDistribution() mean?
	pnl::CMixtureGaussianCPD *cpd2 = pnl::CMixtureGaussianCPD::Create(domains[2], numDomains[2], bnet->GetModelDomain(), tableNode1);
	int parentVal[] = { 0, 0 };  // node0 = 0 & node1 = 0
	cpd2->AllocDistribution(&mean2w00, &cov2w00, 2.0f, NULL, parentVal);
	parentVal[1] = 1;  // node0 = 0 & node1 = 1
	cpd2->AllocDistribution(&mean2w01, &cov2w01, 1.0f, NULL, parentVal);
	parentVal[0] = 1;  // node0 = 1 & node1 = 1
	cpd2->AllocDistribution(&mean2w11, &cov2w11, 1.0f, NULL, parentVal);
	parentVal[1] = 0;  // node0 = 1 & node1 = 0
	cpd2->AllocDistribution(&mean2w10, &cov2w10, 1.0f, NULL, parentVal);
	bnet->AttachFactor(cpd2);

	pnl::CTabularCPD *cpd3 = pnl::CTabularCPD::Create(domains[3], numDomains[3], bnet->GetModelDomain(), tableNode3);
	bnet->AttachFactor(cpd3);

	pnl::CTabularCPD *cpd4 = pnl::CTabularCPD::Create(domains[4], numDomains[4], bnet->GetModelDomain(), tableNode4);
	bnet->AttachFactor(cpd4);

	pnl::CMixtureGaussianCPD *cpd5 = pnl::CMixtureGaussianCPD::Create(domains[5], numDomains[5], bnet->GetModelDomain(), tableNode4);
	parentVal[0] = parentVal[1] = 0;  // node3 = 0 & node4 = 0
	cpd5->AllocDistribution(&mean5w00, &cov5w00, 2.0f, NULL, parentVal);
	parentVal[1] = 1;  // node3 = 0 & node4 = 1
	cpd5->AllocDistribution(&mean5w01, &cov5w01, 1.0f, NULL, parentVal);
	parentVal[0] = 1;  // node3 = 1 & node4 = 1
	cpd5->AllocDistribution(&mean5w11, &cov5w11, 1.0f, NULL, parentVal);
	parentVal[1] = 0;  // node3 = 1 & node4 = 0
	cpd5->AllocDistribution(&mean5w10, &cov5w10, 1.0f, NULL, parentVal);
	bnet->AttachFactor(cpd5);
#endif
#endif

	// create DBN using BNet
	return pnl::CDBN::Create(bnet);
}

// [ref]
//	"Probabilistic Network Library: User Guide and Reference Manual", pp. 2-32 ~ 33
//	learn_dbn_with_ar_gaussian_observations() in dbn_example.cpp
void learn_hmm_with_mixture_of_gaussians_observations(const boost::scoped_ptr<pnl::CDBN> &dbn, const int numTimeSeries)
{
	// define number of slices in the every time series
#if 1
	// [ref] learn_dbn_with_ar_gaussian_observations() in dbn_example.cpp
	pnl::intVector numSlices(numTimeSeries);
	pnl::pnlRand(numTimeSeries, &numSlices.front(), 3, 20);

	//
	const boost::scoped_ptr<const pnl::CBNet> unrolledBNet(static_cast<pnl::CBNet *>(dbn->UnrollDynamicModel(numTimeSeries)));
#else
	// [ref] learn_hmm_with_ar_gaussian_observations() in this file
	const pnl::intVector numSlices(numTimeSeries, 2);
#endif

    std::cout << "***** 111" << std::endl;  // FIXME [delete] >>
#if 1
	// generate evidences in a random way
	pnl::pEvidencesVecVector evidences;
	dbn->GenerateSamples(&evidences, numSlices);

    std::cout << "***** 222" << std::endl;  // FIXME [delete] >>

	// create DBN for learning
	// FIXME [check] >> this implementation isn't verified
	const pnl::CBNet *bnetDBN = dynamic_cast<const pnl::CBNet *>(dbn->GetStaticModel());
	const boost::scoped_ptr<pnl::CDBN> dbnToLearn(pnl::CDBN::Create(pnl::CBNet::Copy(bnetDBN)));

    std::cout << "***** 333" << std::endl;  // FIXME [delete] >>

	// create learning engine
	const boost::scoped_ptr<pnl::CEMLearningEngineDBN> learnEngine(pnl::CEMLearningEngineDBN::Create(dbnToLearn.get()));
#else
	pnl::pEvidencesVecVector evidences;
    std::cout << "***** 444" << std::endl;  // FIXME [delete] >>

	// FIXME [add] >> evidences need to be filled

	// create learning engine
	const boost::scoped_ptr<pnl::CEMLearningEngineDBN> learnEngine(pnl::CEMLearningEngineDBN::Create(dbn.get()));
#endif

    std::cout << "***** 555" << std::endl;  // FIXME [delete] >>

	// set data for learning
	learnEngine->SetData(evidences);

	// start learning
	try
	{
		const float precision = 0.001f;
		const int numMaxIteration = 30;

        std::cout << "***** 666" << std::endl;  // FIXME [delete] >>
		learnEngine->SetTerminationToleranceEM(precision);
		learnEngine->SetMaxIterEM(numMaxIteration);
        std::cout << "***** 777" << std::endl;  // FIXME [delete] >>

		learnEngine->Learn();
        std::cout << "***** 888" << std::endl;  // FIXME [delete] >>
	}
	catch (const pnl::CAlgorithmicException &e)
	{
		std::cout << "fail to learn HMM with mixture-of-Gaussians observations" << e.GetMessage() << std::endl;
		return;
	}

	//
	for (size_t i = 0; i < evidences.size(); ++i)
	{
		for (size_t j = 0; j < evidences[i].size(); ++j)
			delete evidences[i][j];
	}

	// FIXME [implement] >>
}

}  // namespace local
}  // unnamed namespace

void hmm()
{
/*
	// simple HMM
	std::cout << "========== infer MPE in a simple HMM" << std::endl;
	{
		const boost::scoped_ptr<pnl::CDBN> simpleHMM(local::create_simple_hmm());

		if (!simpleHMM)
		{
			std::cout << "fail to create a probabilistic graphical model at " << __LINE__ << " in " << __FILE__ << std::endl;
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

		local::infer_mpe_in_hmm(simpleHMM);
	}

	// HMM with autoregressive Gaussian observations
	std::cout << "\n========== HMM with AR Gaussian observations" << std::endl;
	{
		const boost::scoped_ptr<pnl::CDBN> arHMM(local::create_hmm_with_ar_gaussian_observations());
		//const boost::scoped_ptr<pnl::CDBN> arHMM(pnl::CDBN::Create(pnl::pnlExCreateRndArHMM()));

		if (!arHMM)
		{
			std::cout << "fail to create a probabilistic graphical model at " << __LINE__ << " in " << __FILE__ << std::endl;
			return;
		}

		// get content of Graph
		arHMM->GetGraph()->Dump();

		//
		local::learn_hmm_with_ar_gaussian_observations(arHMM);
	}
*/
	// HMM with mixture-of-Gaussians observations
	std::cout << "\n========== HMM with mixture-of-Gaussians observations" << std::endl;
	{
		const boost::scoped_ptr<pnl::CDBN> hmm(local::create_hmm_with_mixture_of_gaussians_observations());

		if (!hmm)
		{
			std::cout << "fail to create a probabilistic graphical model at " << __LINE__ << " in " << __FILE__ << std::endl;
			return;
		}

		//
		const int numTimeSeries = 500;
		local::learn_hmm_with_mixture_of_gaussians_observations(hmm, numTimeSeries);
	}
}
