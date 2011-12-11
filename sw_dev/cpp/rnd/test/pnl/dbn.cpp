#include "stdafx.h"
#include <pnl_dll.hpp>
#include <iostream>


namespace {
namespace local {

pnl::CDBN * create_model()
{
	// create static model
	const int numNodes = 4;  // number of nodes
	const int numNodeTypes = 1;  // number of node types (all nodes are discrete)

	// 1) first need to specify the graph structure of the model.
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
	
	// 2) create the Model Domain.
	pnl::nodeTypeVector nodeTypes(numNodeTypes);
	nodeTypes[0].SetType(true, 2);

	pnl::intVector nodeAssociation;
	nodeAssociation.assign(numNodes, 0);
	
	pnl::CModelDomain *pMD = pnl::CModelDomain::Create(nodeTypes, nodeAssociation);
	
	// 3) create static BNet with random matrices
	pnl::CBNet *pBNet = pnl::CBNet::CreateWithRandomMatrices(pGraph, pMD);
	
	// 4) create DBN
	pnl::CDBN *pDBN = pnl::CDBN::Create(pBNet);

	return pDBN;
}

void infer_model()
{
#if 1
	pnl::CBNet *pBNetForArHMM = pnl::pnlExCreateRndArHMM();
	pnl::CDBN *pArHMM = pnl::CDBN::Create(pBNetForArHMM);
#else
	pnl::CDBN *pArHMM = create_model();
#endif

	// get content of Graph
	pArHMM->GetGraph()->Dump();
	pnl::CGraph *g = pArHMM->GetGraph();

	// create an inference engine
	pnl::C1_5SliceJtreeInfEngine *pInfEng = pnl::C1_5SliceJtreeInfEngine::Create(pArHMM);

	// number of time slices for unrolling
	const int numTimeSlices = 5;
	const pnl::CPotential *pQueryJPD = NULL;

	// create evidence for every slice
	pnl::CEvidence **pEvidences = new pnl::CEvidence *[numTimeSlices];

	// let node 1 is always observed
	const int numObsNodes[] = { 1 };
	pnl::valueVector obsNodesVals(1);

	for (int i = 0; i < numTimeSlices; ++i)
	{
		// generate random value
		// all nodes in the model are discrete
		obsNodesVals[0].SetInt(std::rand() % 2);
		pEvidences[i] = pnl::CEvidence::Create(pArHMM, 1, numObsNodes, obsNodesVals);
	}

	//---------------------------------------------------------------
	// create smoothing procedure
	pInfEng->DefineProcedure(pnl::ptSmoothing, numTimeSlices);
	// enter created evidences
	pInfEng->EnterEvidence(pEvidences, numTimeSlices);
	// start smoothing process
	pInfEng->Smoothing();

	// choose query set of nodes for every slice
	const int queryPrior[] = { 0 };
	const int queryPriorSize = 1;
	const int query[] = { 0, 2 };
	const int querySize = 2;

	// inference results gaining and representation
	std::cout << " Results of smoothing " << std::endl;
	
	int slice = 0;
	pInfEng->MarginalNodes(queryPrior, queryPriorSize, slice);
	pQueryJPD = pInfEng->GetQueryJPD();

	std::cout << "Query slice" << slice << std::endl;

	int numNodes;
	const int *domain = NULL;
	pQueryJPD->GetDomain(&numNodes, &domain);

	std::cout << " domain :";
	for (int i = 0; i < numNodes; ++i)
	{
		std::cout << domain[i] << " ";
	}
	std::cout << std::endl;

	pnl::CMatrix<float> *pMat = pQueryJPD->GetMatrix(pnl::matTable);

	// graphical model hase been created using dense matrix
	std::cout << " probability distribution" << std::endl;
	int nEl;
	const float *data = NULL;
	static_cast<pnl::CNumericDenseMatrix<float> *>(pMat)->GetRawData(&nEl, &data);
	for (int i = 0; i < nEl; ++i)
	{
		std::cout << " " << data[i];
	}
	std::cout << std::endl;

	for (slice = 1; slice < numTimeSlices; ++slice)
	{
		pInfEng->MarginalNodes(query, querySize, slice);
		pQueryJPD = pInfEng->GetQueryJPD();
		std::cout << "Query slice" << slice << std::endl;
		// representation information using Dump()
		pQueryJPD->Dump();
	}

	slice = 0;

	//---------------------------------------------------------------
	// create filtering procedure
	pInfEng->DefineProcedure(pnl::ptFiltering);
	pInfEng->EnterEvidence(&(pEvidences[slice]), 1);
	pInfEng->Filtering(slice);

	pInfEng->MarginalNodes(queryPrior, queryPriorSize);
	pQueryJPD = pInfEng->GetQueryJPD();

	std::cout << " Results of filtering " << std::endl;
	std::cout << " Query slice " << slice << std::endl;
	pQueryJPD->Dump();
	for (slice = 1; slice < numTimeSlices; ++slice)
	{
		pInfEng->EnterEvidence(&(pEvidences[slice]), 1);
		pInfEng->Filtering(slice);

		pInfEng->MarginalNodes(query, querySize);
		pQueryJPD = pInfEng->GetQueryJPD();

		std::cout << " Query slice " << slice << std::endl;
		pQueryJPD->Dump();
	}

	//---------------------------------------------------------------
	// create fixed-lag smoothing (online)
	const int lag = 2;
	pInfEng->DefineProcedure(pnl::ptFixLagSmoothing, lag);

	for (slice = 0; slice < lag + 1; ++slice)
	{
		pInfEng->EnterEvidence(&(pEvidences[slice]), 1);
	}

	pInfEng->FixLagSmoothing(slice);

	pInfEng->MarginalNodes(queryPrior, queryPriorSize);
	pQueryJPD = pInfEng->GetQueryJPD();

	std::cout << " Results of fixed-lag smoothing " << std::endl;
	std::cout << " Query slice " << slice << std::endl;
	pQueryJPD->Dump();

	std::cout << std::endl;

	for ( ; slice < numTimeSlices; ++slice)
	{
		pInfEng->EnterEvidence(&(pEvidences[slice]), 1);
		pInfEng->FixLagSmoothing(slice);

		pInfEng->MarginalNodes(query, querySize);
		pQueryJPD = pInfEng->GetQueryJPD();

		std::cout << " Query slice " << slice << std::endl;
		pQueryJPD->Dump();
	}
	delete pInfEng;

	for (slice = 0; slice < numTimeSlices; ++slice)
	{
		delete pEvidences[slice];
	}
	delete [] pEvidences;

	delete pArHMM;
}

}  // namespace local
}  // unnamed namespace

void dbn()
{
	local::infer_model();
}
