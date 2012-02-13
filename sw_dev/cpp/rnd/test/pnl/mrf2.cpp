#include "stdafx.h"
#include <pnl_dll.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>


namespace {
namespace local {

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/APearlInfEngineMRF2.cpp
pnl::CMRF2 * create_simple_mrf2_model()
{
	const int nnodes = 11; //4;//7;
	const int numClqs = 10; //3;// 6;

	pnl::nodeTypeVector nodeTypes(1, pnl::CNodeType());
	nodeTypes[0] = pnl::CNodeType(true, 2);

	pnl::intVector nodeAssociation(nnodes, 0);

	pnl::CModelDomain *pMD = pnl::CModelDomain::Create(nodeTypes, nodeAssociation);

	// create graphical model by clqs;
	int *clqSizes  = new int [numClqs];
	for (int i = 0; i < numClqs; ++i)
	{
		clqSizes[i] = 2;
	}

	const int clqs0[] = { 0, 1 };
	const int clqs1[] = { 1, 2 };
	const int clqs2[] = { 1, 3 };
	const int clqs3[] = { 0, 4 };
	const int clqs4[] = { 4, 5 };
	const int clqs5[] = { 4, 6 };
	const int clqs6[] = { 6, 7 };
	const int clqs7[] = { 6, 8 };
	const int clqs8[] = { 6, 9 };
	const int clqs9[] = { 9, 10 };
	const int *clqs[] = { clqs0, clqs1, clqs2, clqs3, clqs4, clqs5, clqs6, clqs7, clqs8, clqs9 };

	pnl::CMRF2 *myModel = pnl::CMRF2::Create(numClqs, clqSizes, clqs, pMD);

	myModel->GetGraph()->Dump();
	// we creates every factor - it is factor

	// number of factors is the same as number of cliques - one per clique
	pnl::CFactor **myParams = new pnl::CFactor * [numClqs];

	// to create factors we need to create their tables

	// create container for Factors
	myModel->AllocFactors();

	// create array of data for every parameter
	const float Data0[] = { 0.6f, 0.4f, 0.8f, 0.2f };
	const float Data1[] = { 0.5f, 0.5f, 0.7f, 0.3f };
	const float Data2[] = { 0.1f, 0.9f, 0.3f, 0.7f };
	const float Data3[] = { 0.1f, 0.9f, 0.2f, 0.8f }; //{ 0.01f, 0.99f, 0.02f, 0.98f };
	const float Data4[] = { 0.2f, 0.8f, 0.3f, 0.7f };
	const float Data5[] = { 0.4f, 0.6f, 0.6f, 0.4f };
	const float Data6[] = { 0.8f, 0.2f, 0.9f, 0.1f };
	const float Data7[] = { 1.0f, 0.0f, 0.0f, 1.0f };
	const float Data8[] = { 0.5f, 0.5f, 0.5f, 0.5f };
	const float Data9[] = { 0.1f, 0.9f, 0.2f, 0.8f };
	const float *Data[] = { Data0, Data1, Data2, Data3, Data4, Data5, Data6, Data7, Data8, Data9 };
	for (int i = 0; i < numClqs; ++i)
	{
		myParams[i] = pnl::CTabularPotential::Create(clqs[i], 2, pMD);
		myParams[i]->AllocMatrix(Data[i], pnl::matTable);

		myModel->AttachFactor((pnl::CTabularPotential *)myParams[i]);
	}

	delete (myParams);  // TODO [check] >.
	delete [] clqSizes;
	delete pMD;

	return myModel;
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/APearlInfEngineMRF2.cpp
void infer_simple_mrf2_using_belief_propagation_algorithm(const boost::scoped_ptr<pnl::CMRF2> &mrf2)
{
	const int nnodes = mrf2->GetNumberOfNodes();  // TODO [check] >> is it correct?
	const float eps = 1e-5f;

	// now we can start inference
	// first we start inference without evidence

	// create evidence
	pnl::CEvidence *myEvid = pnl::CEvidence::Create(mrf2.get(), 0, NULL, pnl::valueVector());

	// now we can compare Naive Inf results with Pearl Inf results
	pnl::CPearlInfEngine *myInfP = pnl::CPearlInfEngine::Create(mrf2.get());
	myInfP->EnterEvidence(myEvid);

	pnl::CNaiveInfEngine *myInfN = pnl::CNaiveInfEngine::Create(mrf2.get());
	myInfN->EnterEvidence(myEvid);

/*
	pnl::CJtreeInfEngine *myInfJ = pnl::CJtreeInfEngine::Create(mrf2.get());
	myInfJ->EnterEvidence(myEvid);
*/

/*
	// data from Matlab
	const float marginals0[] = { 0.5f, 0.5f };
	const float marginals1[] = { 0.7f, 0.3f };
	const float marginals2[] = { 0.56f, 0.44f };
	const float marginals3[] = { 0.16f, 0.84f }; //{ 0.16f, 0.84f };
	const float marginals4[] = { 0.15f, 0.85f }; //{ 0.015f, 0.985f };
	const float marginals5[] = { 0.285f, 0.715f }; //{ 0.2985f, 0.7015f };
	const float marginals6[] = { 0.57f, 0.43f }; //{ 0.597f, 0.403f };
	const float marginals7[] = { 0.843f, 0.157f }; //{ 0.8403f, 0.1597f };
	const float marginals8[] = { 0.57f, 0.43f }; //{ 0.597f, 0.403f };
	const float marginals9[] = { 0.5f, 0.5f };
	const float marginals10[] = { 0.15f, 0.85f };

	const float *marginals[] = {
		marginals0, marginals1, marginals2, marginals3,
		marginals4, marginals5, marginals6, marginals7,
		marginals8, marginals9, marginals10
	};
*/

	const int querySize = 1;
	//int *query = (int *)trsGuardcAlloc(querySize, sizeof(int));
	int *query = new int [querySize];
	const pnl::CPotential *myQueryJPDP;
	const pnl::CPotential *myQueryJPDN;
	//const pnl::CPotential *myQueryJPDJ;
	//pnl::CNumericDenseMatrix<float> *myMatrP = NULL;
	//pnl::CNumericDenseMatrix<float> *myMatrN = NULL;
	//pnl::CNumericDenseMatrix<float> *myMatrJ = NULL;
	//int matPSize;
	//int matNSize;
	//int matJSize;
	for (int i = 0; i < nnodes; ++i)
	{
		query[0] = i;
		myInfP->MarginalNodes(query, querySize);
		myInfN->MarginalNodes(query, querySize);
		//myInfJ->MarginalNodes(query, querySize);

		myQueryJPDP = myInfP->GetQueryJPD();
		myQueryJPDN = myInfN->GetQueryJPD();
		if (!myQueryJPDP->IsFactorsDistribFunEqual(myQueryJPDN, eps, 0))
		{
			// TODO [implement] >>
		}
	}

	// we can add evidence and compare results with results from NaiveInfrenceEngine
	int numObsNodes = 0;
	while ((numObsNodes < 1) || (numObsNodes > nnodes))
	{
		//trsiRead( &numObsNodes, "1", "number of observed nodes");
	}
	int seed1 = 1021450643; //pnlTestRandSeed();
	// create string to display the value
	char *value = new char [20];
#if 0
	_itoa(seed1, value, 10);
#else
	sprintf(value, "%d", seed1);
#endif
	//trsiRead(&seed1, value, "Seed for srand to define NodeTypes etc.");
	//trsWrite(TW_CON|TW_RUN|TW_DEBUG|TW_LST, "seed for rand = %d\n", seed1);
	std::srand(seed1);

	//int *ObsNodes = (int *)trsGuardcAlloc(numObsNodes, sizeof(int));
	int *ObsNodes = new int [numObsNodes];
	pnl::valueVector ObsValues;
	ObsValues.assign(numObsNodes, pnl::Value(0));
	pnl::intVector residuaryNodesFor ;
	for (int i = 0; i < nnodes; ++i)
	{
		residuaryNodesFor.push_back(i);
	}
	for (int i = 0; i < numObsNodes; ++i)
	{
		const int j = std::rand() % (nnodes - i);
		ObsNodes[i] = residuaryNodesFor[j];
		residuaryNodesFor.erase(residuaryNodesFor.begin() + j);
		ObsValues[i].SetInt(std::rand() % (mrf2->GetNodeType(ObsNodes[i])->GetNodeSize()));
	}

	residuaryNodesFor.clear();
	pnl::CEvidence *myEvidWithObs = pnl::CEvidence::Create(mrf2.get(), numObsNodes, ObsNodes, ObsValues);
	myInfP->EnterEvidence(myEvidWithObs);
	//NumIters = myInfP->GetNumberOfProvideIterations();
	myInfN->EnterEvidence(myEvidWithObs);
	for (int i = 0; i < nnodes; ++i)
	{
		query[0] = i;
		myInfP->MarginalNodes(query, querySize);
		myInfN->MarginalNodes(query, querySize);
		myQueryJPDP = myInfP->GetQueryJPD();
		myQueryJPDN = myInfN->GetQueryJPD();
		if (!myQueryJPDP->IsFactorsDistribFunEqual(myQueryJPDN, eps, 0))
		{
			// TODO [implement] >>
		}
	}

	//pnl::CPearlInfEngine::Release(&myInfP);
	delete (myInfP);
	delete (myInfN);
	//delete (myInfJ);
	delete (myEvidWithObs);
	delete (myEvid);

	delete [] query;
	delete [] ObsNodes;
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/AGaussianMRF2.cpp
pnl::CMRF2 * create_gaussian_mrf2_1()
{
	std::ofstream logFileStream("logForPearlandFG.txt", std::ios::app);
	pnl::LogDrvStream logDriver(&logFileStream, pnl::eLOG_ALL, pnl::eLOGSRV_ALL);

	pnl::CNodeType *nodeTypes = new pnl::CNodeType [3];
	for (int i = 0; i < 3; ++i)
	{
		nodeTypes[i].SetType(false, i + 1);
	}

	// different information below
	const int smallNumNodes = 3;
	const int smallNumClqs = 2;

	const int smallClqSizes[] = { 2, 2 };

	const int SmallClqs0[2] = { 0, 1 };
	const int SmallClqs1[2] = { 1, 2 };
	const int *SmallClqs[] = { SmallClqs0, SmallClqs1 };

	const int simplestNodeAssociation[3] = { 0, 0, 0 };

	pnl::CMRF2 *simplestSmallModel = pnl::CMRF2::Create(smallNumNodes, 3, nodeTypes, simplestNodeAssociation, smallNumClqs, smallClqSizes, SmallClqs);
	pnl::CModelDomain *pMD = simplestSmallModel->GetModelDomain();

	simplestSmallModel->GetGraph()->Dump();
	pnl::CFactor **mySmallParams = new pnl::CFactor * [smallNumClqs];
	const float simDataM0[] = { 1.0f, 2.0f };
	const float simDataM1[] = { 4.0f, 3.0f };
	const float *simDataM[] = { simDataM0, simDataM1 };

	const float simDataCov0[] = { 3.0f, 3.0f, 3.0f, 4.0f };
	const float simDataCov1[] = { 1.0f, 1.0f, 1.0f, 3.0f };
	const float *simDataCov[] = { simDataCov0, simDataCov1 };

	const float simNorms[] = { 1.0f, 1.0f };

	simplestSmallModel->AllocFactors();

	pnl::CFactorGraph *pSimplestFG = pnl::CFactorGraph::Create(simplestSmallModel->GetModelDomain(), smallNumClqs);

	const pnl::CNodeType *domainNT[2];
	for (int i = 0; i < smallNumClqs; ++i)
	{
		for (int j = 0; j < 2; ++j)
		{
			domainNT[j] = simplestSmallModel->GetNodeType(SmallClqs[i][j]);
		}

		mySmallParams[i] = pnl::CGaussianPotential::Create(SmallClqs[i], 2, pMD);
		static_cast<pnl::CGaussianPotential *>(mySmallParams[i])->SetCoefficient(simNorms[i], 1);
		mySmallParams[i]->AllocMatrix(simDataM[i], pnl::matMean);
		mySmallParams[i]->AllocMatrix(simDataCov[i], pnl::matCovariance);
		simplestSmallModel->AttachFactor((pnl::CGaussianPotential *)mySmallParams[i]);
		pSimplestFG->AllocFactor(2, SmallClqs[i]);

		pnl::pFactorVector factors;
		pSimplestFG->GetFactors(2, SmallClqs[i], &factors);
		static_cast<pnl::CGaussianPotential *>(factors[0])->SetCoefficient(simNorms[i], 1);
		factors[0]->AllocMatrix(simDataM[i], pnl::matMean);
		factors[0]->AllocMatrix(simDataCov[i], pnl::matCovariance);
	}

	delete [] nodeTypes;
	delete [] mySmallParams;
	delete pSimplestFG;

	return simplestSmallModel;
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/AGaussianMRF2.cpp
void infer_gaussian_mrf2_using_inference_algorithm_1(const boost::scoped_ptr<pnl::CMRF2> &mrf2)
{
	const int smallNumNodes = mrf2->GetNumberOfNodes();  // TODO [check] >> is it correct?
	const float eps = 1e-4f;

	pnl::CFactorGraph *pSimplestFG = pnl::CFactorGraph::ConvertFromMNet(mrf2.get());  // TODO [check] >> is it correct?

	const pnl::CPotential *jpdNSm = NULL;
	const pnl::CPotential *jpdPSm = NULL;
	const pnl::CPotential *jpdFGSm = NULL;
	const pnl::CPotential *jpdJSm = NULL;
	int isTheSame;

	pnl::CEvidence *simEmptyEv = pnl::CEvidence::Create(mrf2.get(), 0, NULL, pnl::valueVector());
	pnl::CNaiveInfEngine *iSimNaiveEmpt = pnl::CNaiveInfEngine::Create(mrf2.get());
	iSimNaiveEmpt->EnterEvidence(simEmptyEv);

	pnl::CFGSumMaxInfEngine *iSimFGInfEng = pnl::CFGSumMaxInfEngine::Create(pSimplestFG);
	iSimFGInfEng->EnterEvidence(simEmptyEv);

	pnl::CInfEngine *iSimJtreeEmpt = pnl::CJtreeInfEngine::Create(mrf2.get());
	iSimJtreeEmpt->EnterEvidence(simEmptyEv);

	pnl::CPearlInfEngine *iSimPearlEmpt = pnl::CPearlInfEngine::Create(mrf2.get());
	iSimPearlEmpt->EnterEvidence(simEmptyEv);

	int query = 0;
	for (int i = 0; i < smallNumNodes; ++i)
	{
		query = i;
		iSimNaiveEmpt->MarginalNodes(&query, 1);
		jpdNSm = iSimNaiveEmpt->GetQueryJPD();

		iSimPearlEmpt->MarginalNodes(&query, 1);
		jpdPSm = iSimPearlEmpt->GetQueryJPD();

		iSimJtreeEmpt->MarginalNodes(&query, 1);
		jpdJSm = iSimJtreeEmpt->GetQueryJPD();

		iSimFGInfEng->MarginalNodes(&query, 1);
		jpdFGSm = iSimFGInfEng->GetQueryJPD();

		isTheSame = jpdPSm->IsFactorsDistribFunEqual(static_cast<const pnl::CFactor *>(jpdNSm), eps, 0);
		if (!isTheSame)
		{
			// TODO [implement] >>
		}

		isTheSame = jpdFGSm->IsFactorsDistribFunEqual( jpdPSm, eps, 0  );
		if (!isTheSame)
		{
			// TODO [implement] >>
		}

		isTheSame = jpdJSm->IsFactorsDistribFunEqual(static_cast<const pnl::CFactor *>(jpdNSm), eps, 0);
		if (!isTheSame)
		{
			// TODO [implement] >>
		}
		isTheSame = jpdJSm->IsFactorsDistribFunEqual(static_cast<const pnl::CFactor *>(jpdPSm), eps, 0);
		if (!isTheSame)
		{
			// TODO [implement] >>
		}
	}

	int obsNode = 1;
	pnl::valueVector obsVal;
	obsVal.assign(1, pnl::Value(0));
	obsVal[0].SetFlt(3.0f);
	pnl::CEvidence *simOneEv = pnl::CEvidence::Create(mrf2.get(), 1, &obsNode, obsVal);

	pnl::CNaiveInfEngine *iSimNaiveOne = pnl::CNaiveInfEngine::Create(mrf2.get());
	iSimNaiveOne->EnterEvidence(simOneEv);

	pnl::CPearlInfEngine *iSimPearlOne = pnl::CPearlInfEngine::Create(mrf2.get());
	iSimPearlOne->SetMaxNumberOfIterations(10);
	iSimPearlOne->EnterEvidence(simOneEv);

	pnl::CFGSumMaxInfEngine *iSimFGOne = pnl::CFGSumMaxInfEngine::Create(pSimplestFG);
	iSimFGOne->EnterEvidence(simOneEv);

	pnl::CJtreeInfEngine *iSimJtreeOne = pnl::CJtreeInfEngine::Create(mrf2.get());
	iSimJtreeOne->EnterEvidence(simOneEv);

	query = 0;
	for (int i = 0; i < smallNumNodes; ++i)
	{
		query = i;
		iSimNaiveOne->MarginalNodes(&query, 1);
		jpdNSm = iSimNaiveOne->GetQueryJPD();

		iSimPearlOne->MarginalNodes(&query, 1);
		jpdPSm = iSimPearlOne->GetQueryJPD();

		iSimJtreeOne->MarginalNodes(&query, 1);
		jpdJSm = iSimJtreeOne->GetQueryJPD();

		iSimFGOne->MarginalNodes(&query, 1);
		jpdFGSm = iSimFGOne->GetQueryJPD();

		isTheSame = jpdPSm->IsFactorsDistribFunEqual(static_cast<const pnl::CFactor *>(jpdNSm), eps, 0);
		if (!isTheSame)
		{
			// TODO [implement] >>
		}

		isTheSame = jpdFGSm->IsFactorsDistribFunEqual(jpdPSm, eps, 0);
		if (!isTheSame)
		{
			// TODO [implement] >>
		}

		isTheSame = jpdFGSm->IsFactorsDistribFunEqual(static_cast<const  pnl::CFactor *>(jpdNSm), eps, 0);
		if (!isTheSame)
		{
			// TODO [implement] >>
		}
		isTheSame = jpdJSm->IsFactorsDistribFunEqual(static_cast<const  pnl::CFactor *>(jpdNSm), eps, 0);
		if (!isTheSame)
		{
			// TODO [implement] >>
		}
	}

	delete simEmptyEv;
	delete iSimPearlOne;
	delete iSimJtreeOne;
	delete iSimNaiveOne;
	//pnl::CPearlInfEngine::Release(&iSimPearlEmpt);
	delete iSimPearlEmpt;
	delete iSimJtreeEmpt;
	delete iSimFGInfEng;
	delete iSimNaiveEmpt;
	delete iSimFGOne;
	//pnl::CPearlInfEngine::Release(&iSimPearlOne);
	delete simOneEv;

	delete pSimplestFG;
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/AGaussianMRF2.cpp
pnl::CMRF2 * create_gaussian_mrf2_2()
{
	pnl::CNodeType *nodeTypes = new pnl::CNodeType [3];
	for (int i = 0; i < 3; ++i)
	{
		nodeTypes[i].SetType(false, i + 1);
	}

	// different information below
	const int smallNumNodes = 3;
	const int smallNumClqs = 2;

	const int smallClqSizes[] = { 2, 2 };

	const int SmallClqs0[2] = { 0, 1 };
	const int SmallClqs1[2] = { 1, 2 };
	const int *SmallClqs[] = { SmallClqs0, SmallClqs1 };

	const int simplestNodeAssociation[3] = { 0, 0, 0 };
	const int smallNodeAssociation[3] = { 1, 2, 0 };
	
	pnl::CMRF2 *simplestSmallModel = pnl::CMRF2::Create(smallNumNodes, 3, nodeTypes, simplestNodeAssociation, smallNumClqs, smallClqSizes, SmallClqs);
	pnl::CMRF2 *SmallModel = pnl::CMRF2::Create(smallNumNodes, 3, nodeTypes, smallNodeAssociation, smallNumClqs, smallClqSizes, SmallClqs);
	SmallModel->GetGraph()->Dump();

	const float smDataM0[] = { 0.6f, 0.4f, 1.3f, 1.7f, 1.9f };
	const float smDataM1[] = { 1.6f, 1.7f, 1.8f, 2.1f };
	const float *smDataM[] = { smDataM0, smDataM1 };

	const float smDataCov0[] = { 7.4f, 7.5f, 7.6f, 7.4f, 7.3f, 7.5f, 7.2f, 7.3f, 7.3f, 7.5f, 7.6f, 7.3f, 7.8f, 7.1f, 7.1f, 7.4f, 7.3f, 7.1f, 7.1f, 7.6f, 7.3f, 7.5f, 7.1f, 7.6f, 7.3f  };
	//const float smDataCov0[] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0, 1.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.3f  };
	const float smDataCov1[] = { 3.0f, 4.0f, 5.0f, 6.0f, 4.0f, 8.0f, 9.0f, 1.0f, 5.0f, 9.0f, 3.0f, 4.0f, 6.0f, 1.0f, 4.0f, 8.0f };
	//const float smDataCov1[] = { 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f };
	const float *smDataCov[] = { smDataCov0, smDataCov1 };

	const float smNorms[] = { 1.0f, 1.0f };

	SmallModel->AllocFactors();
	pnl::CModelDomain *pMD1 = SmallModel->GetModelDomain();

	pnl::CFactors *pFactors = pnl::CFactors::Create(smallNumClqs);

	pnl::CFactor **mySmallParams1 = new pnl::CFactor * [smallNumClqs];
	const pnl::CNodeType *domainNT[2];
	for (int i = 0; i < smallNumClqs; ++i)
	{
		for (int j = 0; j < 2; ++j)
		{
			domainNT[j] = SmallModel->GetNodeType(SmallClqs[i][j]);
		}
		mySmallParams1[i] = pnl::CGaussianPotential::Create(SmallClqs[i], 2, pMD1);
		static_cast<pnl::CGaussianPotential *>(mySmallParams1[i])->SetCoefficient(smNorms[i], 0);
		mySmallParams1[i]->AllocMatrix(smDataM[i], pnl::matMean);
		mySmallParams1[i]->AllocMatrix(smDataCov[i], pnl::matCovariance);

		SmallModel->AttachFactor((pnl::CGaussianPotential *)mySmallParams1[i]);
		pFactors->AddFactor(mySmallParams1[i]->Clone());
	}

	delete [] nodeTypes;
	delete [] mySmallParams1;
	delete pFactors;

	return SmallModel;
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/AGaussianMRF2.cpp
void infer_gaussian_mrf2_using_inference_algorithm_2(const boost::scoped_ptr<pnl::CMRF2> &mrf2)
{
	const int smallNumNodes = 3;
	const float eps = 1e-5f;

	const pnl::CPotential *jpdNSm = NULL;
	const pnl::CPotential *jpdPSm = NULL;
	const pnl::CPotential *jpdFGSm = NULL;
	const pnl::CPotential *jpdJSm = NULL;
	int isTheSame;

	pnl::CEvidence *smallEmptyEv = pnl::CEvidence::Create(mrf2.get(), 0, NULL, pnl::valueVector());

	// TODO [check] >> is it correct?
	pnl::CFactorGraph *smallFG = pnl::CFactorGraph::ConvertFromMNet(mrf2.get());
	// create the factor graph based on factors
	//pnl::CFactorGraph *smallFG = pnl::CFactorGraph::Create(mrf2->GetModelDomain(), pFactors);

	pnl::CFGSumMaxInfEngine *iSmFGInfEmpt = pnl::CFGSumMaxInfEngine::Create(smallFG);
	iSmFGInfEmpt->EnterEvidence(smallEmptyEv);

	pnl::CInfEngine *iSmNaiveEmpt = pnl::CNaiveInfEngine::Create(mrf2.get());
	iSmNaiveEmpt->EnterEvidence(smallEmptyEv);

	pnl::CInfEngine *iSmPearlEmpt = pnl::CPearlInfEngine::Create(mrf2.get());
	iSmPearlEmpt->EnterEvidence(smallEmptyEv);

	int query = 0;
	for (int i = 0; i < smallNumNodes; ++i)
	{
		query = i;
		iSmFGInfEmpt->MarginalNodes(&query, 1);
		jpdFGSm = iSmFGInfEmpt->GetQueryJPD();
		iSmNaiveEmpt->MarginalNodes(&query, 1);
		jpdNSm = iSmNaiveEmpt->GetQueryJPD();
		iSmPearlEmpt->MarginalNodes(&query, 1);
		jpdPSm = iSmPearlEmpt->GetQueryJPD();

		isTheSame = jpdFGSm->IsFactorsDistribFunEqual(static_cast<const pnl::CFactor *>(jpdNSm), eps, 0);
		if (!isTheSame)
		{
			// TODO [implement] >>
		}
		isTheSame = jpdPSm->IsFactorsDistribFunEqual(static_cast<const pnl::CFactor *>(jpdNSm), eps, 0);
		if (!isTheSame)
		{
			// TODO [implement] >>
		}
	}

	int obsNode = 1;
	pnl::valueVector vals(3, pnl::Value(0));
	vals[0].SetFlt(1.0f);
	vals[1].SetFlt(2.0f);
	vals[2].SetFlt(3.0f);
	pnl::CEvidence *smallOneEv = pnl::CEvidence::Create(mrf2.get(), 1, &obsNode, vals);

	pnl::CFGSumMaxInfEngine *iSmFGInfOne = pnl::CFGSumMaxInfEngine::Create(smallFG);
	iSmFGInfOne->EnterEvidence(smallOneEv);

	pnl::CInfEngine *iSmNaiveOne = pnl::CNaiveInfEngine::Create(mrf2.get());
	iSmNaiveOne->EnterEvidence(smallOneEv);

	pnl::CInfEngine *iSmPearlOne = pnl::CPearlInfEngine::Create(mrf2.get());
	iSmPearlOne->EnterEvidence(smallOneEv);

	for (int i = 0; i < smallNumNodes; ++i)
	{
		query = i;
		iSmFGInfOne->MarginalNodes(&query, 1);
		jpdFGSm = iSmFGInfOne->GetQueryJPD();
		iSmNaiveOne->MarginalNodes(&query, 1);
		jpdNSm = iSmNaiveOne->GetQueryJPD();
		iSmPearlOne->MarginalNodes(&query, 1);
		jpdPSm = iSmPearlOne->GetQueryJPD();

		isTheSame = jpdPSm->IsFactorsDistribFunEqual(static_cast<const pnl::CFactor *>(jpdNSm), eps, 0);
		if (!isTheSame)
		{
			// TODO [implement] >>
		}
		isTheSame = jpdFGSm->IsFactorsDistribFunEqual(static_cast<const pnl::CFactor *>(jpdNSm), eps, 0);
		if (!isTheSame)
		{
			// TODO [implement] >>
		}
	}

	delete iSmFGInfOne;
	delete iSmNaiveOne;
	delete iSmPearlOne;
	delete smallOneEv;

	delete iSmFGInfEmpt;
	delete iSmNaiveEmpt;
	delete iSmPearlEmpt;
	delete smallEmptyEv;

	delete smallFG;
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/AGaussianMRF2.cpp
pnl::CMRF2 * create_gaussian_mrf2_3()
{
	const int nnodes = 9;
	const int numClqs =8;

	pnl::CNodeType *nodeTypes = new pnl::CNodeType [3];
	for (int i = 0; i < 3; ++i)
	{
		nodeTypes[i].SetType(false, i + 1);
	}

	const int nodeAssociation[9] = { 0, 0, 1, 2, 0, 1, 1, 2, 0 };

	// create graphical model by clqs;
	int *clqSizes  = new int [numClqs];
	for (int i = 0; i < numClqs; ++i)
	{
		clqSizes[i] = 2;
	}

	const int clqs0[] = { 0, 1 };
	const int clqs1[] = { 1, 2 };
	const int clqs2[] = { 1, 3 };
	const int clqs3[] = { 0, 4 };
	const int clqs4[] = { 4, 5 };
	const int clqs5[] = { 4, 6 };
	const int clqs6[] = { 6, 7 };
	const int clqs7[] = { 6, 8 };
	const int *clqs[] = { clqs0, clqs1, clqs2, clqs3, clqs4, clqs5, clqs6, clqs7 };

	pnl::CMRF2 *myModel = pnl::CMRF2::Create(nnodes, 3, nodeTypes, nodeAssociation, numClqs, clqSizes, clqs);

	myModel->GetGraph()->Dump();

	// we creates every factor - it is factor

	// number of factors is the same as number of cliques - one per clique
	pnl::CFactor **myParams = new pnl::CFactor * [numClqs];

	// to create factors we need to create their tables

	// create container for Factors
	myModel->AllocFactors();
	// create array of data for every parameter
	const float DataM0[] = { 0.6f, 0.4f };
	const float DataM1[] = { 1.5f, 1.5f, 1.7f };
	const float DataM2[] = { 2.1f, 2.9f, 2.3f, 2.7f };
	const float DataM3[] = { 3.1f, 3.9f };
	const float DataM4[] = { 4.2f, 4.8f, 4.3f };
	const float DataM5[] = { 5.4f, 5.6f, 5.6f };
	const float DataM6[] = { 6.8f, 6.2f, 6.9f, 6.1f, 6.7f };
	const float DataM7[] = { 7.1f, 7.4f, 7.3f };
	const float *DataM[] = { DataM0, DataM1, DataM2, DataM3, DataM4, DataM5, DataM6, DataM7 };

	const float DataC0[] = { 1.0f, 3.0f, 3.0f, 4.0f };
	//const float DataC0[] = { 1.0f, 0.0f, 0.0f, 4.0f };
	const float DataC1[] = { 2.0f, 3.0f, 4.0f, 3.0f, 6.0f, 7.0f, 4.0f, 7.0f, 1.0f };
	//const float DataC1[] = { 2.0f, 0.0f, 0.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f, 1.0f };
	const float DataC2[] = { 3.0f, 4.0f, 5.0f, 6.0f, 4.0f, 8.0f, 9.0f, 1.0f, 5.0f, 9.0f, 3.0f, 4.0f, 6.0f, 1.0f, 4.0f, 8.0f };
	//const float DataC2[] = { 3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 8.0f };
	const float DataC3[] = { 4.0f, 6.0f, 6.0f, 7.0f };
	//const float DataC3[] = { 4.0f, 0.0f, 0.0f, 7.0f };
	const float DataC4[] = { 5.0f, 6.0f, 7.0f, 6.0f, 9.0f, 1.0f, 7.0f, 1.0f, 4.0f };
	//const float DataC4[] = { 5.0f, 0.0f, 0.0f, 0.0f, 9.0f, 0.0f, 0.0f, 0.0f, 4.0f };
	const float DataC5[] = { 6.0f, 7.0f, 8.0f, 7.0f, 1.0f, 2.0f, 8.0f, 2.0f, 5.0f };
	//const float DataC5[] = { 6.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 5.0f };
	const float DataC6[] = { 7.4f, 7.5f, 7.6f, 7.4f, 7.3f, 7.5f, 7.2f, 7.3f, 7.3f, 7.5f, 7.6f, 7.3f, 7.8f, 7.1f, 7.1f, 7.4f, 7.3f, 7.1f, 7.1f, 7.6f, 7.3f, 7.5f, 7.1f, 7.6f, 7.3f  };
	//const float DataC6[] = { 7.4f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 7.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 7.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 7.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 7.3f  };
	const float DataC7[] = { 8.0f, 9.0f, 1.0f, 9.0f, 3.0f, 4.0f, 1.0f, 4.0f, 7.0f };
	//const float DataC7[] = { 8.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 7.0f    };
	const float *DataC[] = { DataC0, DataC1, DataC2, DataC3, DataC4, DataC5, DataC6, DataC7 };

	const float norms[] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

	pnl::CModelDomain *pMD2 = myModel->GetModelDomain();

	for (int i = 0; i < numClqs; ++i)
	{
		myParams[i] = pnl::CGaussianPotential::Create(clqs[i], 2, pMD2);
		static_cast<pnl::CGaussianPotential *>(myParams[i])->SetCoefficient(norms[i], 1);
		myParams[i]->AllocMatrix(DataM[i], pnl::matH);
		myParams[i]->AllocMatrix(DataC[i], pnl::matK);

		myModel->AttachFactor((pnl::CGaussianPotential *)myParams[i]);
	}

	delete [] nodeTypes;
	delete [] myParams;
	delete [] clqSizes;

	return myModel;
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/AGaussianMRF2.cpp
void infer_gaussian_mrf2_using_inference_algorithm_3(const boost::scoped_ptr<pnl::CMRF2> &mrf2)
{
	const int nnodes = mrf2->GetNumberOfNodes();  // TODO [check] >> is it correct?

	pnl::CEvidence *emptyEv = pnl::CEvidence::Create(mrf2.get(), 0, NULL, pnl::valueVector());

	//start NaiveInf
	pnl::CInfEngine *iNaiveEmpt = pnl::CNaiveInfEngine::Create(mrf2.get());
	iNaiveEmpt->EnterEvidence(emptyEv);

	pnl::CInfEngine *iPearlEmpt = pnl::CPearlInfEngine::Create(mrf2.get());
	iPearlEmpt->EnterEvidence(emptyEv);

	pnl::CFactorGraph *myModelFG = pnl::CFactorGraph::ConvertFromMNet(mrf2.get());

	pnl::CInfEngine *iFGEmpt = pnl::CFGSumMaxInfEngine::Create(myModelFG);
	iFGEmpt->EnterEvidence(emptyEv);

	const pnl::CPotential *jpdN = NULL;
	const pnl::CPotential *jpdP = NULL;
	const pnl::CPotential* jpdFG = NULL;
	//pnl::CNumericDenseMatrix<float> *matMeanN = NULL;
	//pnl::CNumericDenseMatrix<float> *matCovN = NULL;
	//pnl::CNumericDenseMatrix<float> *matMeanP = NULL;
	//pnl::CNumericDenseMatrix<float> *matCovP = NULL;
	//int flagSpecificP;
	//int flagSpecificN;
	//int flagMomN;
	//int flagCanN;
	//int flagMomP;
	//int flagcanP;
	int query = 0;
	int isTheSame;
	const float eps = 1e-3f;
	for (int i = 0; i < nnodes; ++i)
	{
		query = i;
		iNaiveEmpt->MarginalNodes(&query, 1);
		jpdN = iNaiveEmpt->GetQueryJPD();
		iPearlEmpt->MarginalNodes(&query, 1);
		jpdP = iPearlEmpt->GetQueryJPD();
		iFGEmpt->MarginalNodes(&query, 1);
		jpdFG = iFGEmpt->GetQueryJPD();

		isTheSame = jpdP->IsFactorsDistribFunEqual(static_cast<const pnl::CFactor *>(jpdFG), eps, 0);
		if (!isTheSame)
		{
			// TODO [implement] >>
		}
		isTheSame = jpdN->IsFactorsDistribFunEqual(static_cast<const pnl::CFactor *>(jpdFG), eps, 0);
		if (!isTheSame)
		{
			// TODO [implement] >>
		}
	}

	// TODO [check] >> is it correct?
	pnl::CFactorGraph *smallFG = pnl::CFactorGraph::ConvertFromMNet(mrf2.get());
	// create the factor graph based on factors
	//pnl::CFactorGraph *smallFG = pnl::CFactorGraph::Create(mrf2->GetModelDomain(), pFactors);

	// check some methods of FactorGraph
	pnl::CFactorGraph *smallFGCopy = pnl::CFactorGraph::Copy(smallFG);
	const int isValid = smallFGCopy->IsValid();
	if (!isValid)
	{
		// TODO [implement] >>
	}

	delete smallFGCopy;
	delete smallFG;

	//release all memory
	delete myModelFG;

	delete emptyEv;
	delete iNaiveEmpt;
	delete iPearlEmpt;
	delete iFGEmpt;
}

}  // namespace local
}  // unnamed namespace

void mrf2()
{
	// simple pairwise MRF
	std::cout << "========== simple pairwise MRF" << std::endl;
	{
		const boost::scoped_ptr<pnl::CMRF2> mrf2(local::create_simple_mrf2_model());

		if (!mrf2)
		{
			std::cout << "can't create a probabilistic graphical model" << std::endl;
			return;
		}

		// belief propagation (Pearl inference) algorithm
		//local::infer_simple_mrf2_using_belief_propagation_algorithm(mrf2);  // to be corrected
	}

	// Gaussian pairwise MRF
	std::cout << "\n========== Gaussian pairwise MRF" << std::endl;
	{
		const boost::scoped_ptr<pnl::CMRF2> mrf2_1(local::create_gaussian_mrf2_1());
		const boost::scoped_ptr<pnl::CMRF2> mrf2_2(local::create_gaussian_mrf2_2());
		const boost::scoped_ptr<pnl::CMRF2> mrf2_3(local::create_gaussian_mrf2_3());

		if (!mrf2)
		{
			std::cout << "can't create a probabilistic graphical model" << std::endl;
			return;
		}

		// belief propagation (Pearl inference) algorithm
		//local::infer_gaussian_mrf2_using_inference_algorithm_1(mrf2_1);  // to be corrected
		//local::infer_gaussian_mrf2_using_inference_algorithm_2(mrf2_2);  // to be corrected
		//local::infer_gaussian_mrf2_using_inference_algorithm_3(mrf2_3);  // to be corrected
	}
}
