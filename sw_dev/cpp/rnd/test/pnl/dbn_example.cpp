#include "stdafx.h"
#include <pnl_dll.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>


namespace {
namespace local {

pnl::CDBN * create_model()
{
	// create static model
	const int numNodes = 4;  // number of nodes
	const int numNodeTypes = 1;  // number of node types (all nodes are discrete)

	// For the transition network, nodes in the query with the number from 0 to N-1 belong to the time-slice ts-1
	// and nodes with the number from N to 2N-1 belong to the time-slice ts.
	// For the prior time-slice, for example, ts=0, nodes in the query must have numbers from 0 to N-1.
	// [ref] "Probabilistic Network Librayr: User Guide and Reference Manual", p. 2-8 & p. 3-258
	//
	// for the case shown below
	//	-. N = 2
	//	-. for the prior network
	//		nodes belonging to the prior time-slice, ts=0 ==> 0 & 1
	//	-. for the transition network
	//		nodes belonging to the time-slice, ts-1 ==> 0 & 1
	//		nodes belonging to the time-slice, ts ==> 2 & 3

	// 1) first need to specify the graph structure of the model.
	const int numNeighbors[] = { 2, 2, 2, 2 };
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
	
	pnl::CGraph *pGraph = pnl::CGraph::Create(numNodes, numNeighbors, neighs, orients);
	
	// 2) create the Model Domain.
	pnl::nodeTypeVector nodeTypes(numNodeTypes);
	nodeTypes[0].SetType(true, 2);

	pnl::intVector nodeAssociation;
	nodeAssociation.assign(numNodes, 0);
	
	pnl::CModelDomain *pMD = pnl::CModelDomain::Create(nodeTypes, nodeAssociation);
	
	// 3) create static BNet with random matrices.
	pnl::CBNet *pBNet = pnl::CBNet::CreateWithRandomMatrices(pGraph, pMD);
	
	// 4) create DBN.
	pnl::CDBN *pDBN = pnl::CDBN::Create(pBNet);

	return pDBN;
}

void smoothing(pnl::CDBN *pDBN)
{
	// number of time slices for unrolling
	const int numTimeSlices = 5;

	// let node 1 be always observed
	const int obsNodes[] = { 1 };  // 1st node ==> observed node
	pnl::valueVector obsNodesVals(1);

	// create evidence for every time-slice
	pnl::CEvidence **pEvidences = new pnl::CEvidence *[numTimeSlices];
	for (int time_slice = 0; time_slice < numTimeSlices; ++time_slice)
	{
		// generate random value
		// all nodes in the model are discrete
		obsNodesVals[0].SetInt(std::rand() % 2);
		pEvidences[time_slice] = pnl::CEvidence::Create(pDBN, 1, obsNodes, obsNodesVals);
	}

	// create an inference engine
	pnl::C1_5SliceJtreeInfEngine *pInfEng = pnl::C1_5SliceJtreeInfEngine::Create(pDBN);

	// create smoothing procedure
	pInfEng->DefineProcedure(pnl::ptSmoothing, numTimeSlices);
	// enter created evidences
	pInfEng->EnterEvidence(pEvidences, numTimeSlices);
	// start smoothing process
	pInfEng->Smoothing();

	// inference results gaining and representation
	std::cout << "========== Results of smoothing " << std::endl;
	
	// choose query set of nodes for every time-slice
	// for the prior network
	{
		const int queryPrior[] = { 0 };  // 0th node ==> hidden state
		const int queryPriorSize = 1;

		// for the prior time-slice, ts = 0
		int time_slice = 0;
		// calculate joint probability distribution (JPD) & most probability explanation (MPE) (???)
		pInfEng->MarginalNodes(queryPrior, queryPriorSize, time_slice);

		const pnl::CPotential *pQueryJPD = pInfEng->GetQueryJPD();

		std::cout << ">>> Query time-slice: " << time_slice << std::endl;

		int numNodes = 0;
		const int *domain = NULL;
		pQueryJPD->GetDomain(&numNodes, &domain);

		std::cout << " domain: ";
		for (int i = 0; i < numNodes; ++i)
		{
			std::cout << domain[i] << " ";
		}
		std::cout << std::endl;

		pnl::CMatrix<float> *pMat = pQueryJPD->GetMatrix(pnl::matTable);
		const pnl::EMatrixClass &type = pMat->GetMatrixClass();
		if (!(pnl::mcDense == type || pnl::mcNumericDense == type || pnl::mc2DNumericDense == type))
		{
			assert(0);
		}

		// graphical model has been created using dense matrix
		int numData = 0;
		const float *data = NULL;
		dynamic_cast<pnl::CDenseMatrix<float> *>(pMat)->GetRawData(&numData, &data);
		std::cout << " probability distribution: " << std::endl;
		for (int i = 0; i < numData; ++i)
		{
			std::cout << " " << data[i];
		}
		std::cout << std::endl;
	}

	// for the transition network
	{
		const int query[] = { 0, 2 };  // 0th & 2nd nodes ==> hidden states
		const int querySize = 2;

		for (int time_slice = 1; time_slice < numTimeSlices; ++time_slice)
		{
			pInfEng->MarginalNodes(query, querySize, time_slice);

			const pnl::CPotential *pQueryJPD = pInfEng->GetQueryJPD();

			std::cout << ">>> Query time-slice: " << time_slice << std::endl;
			// representation information using Dump()
			pQueryJPD->Dump();
		}
	}

	//
	delete pInfEng;
	for (int time_slice = 0; time_slice < numTimeSlices; ++time_slice)
	{
		delete pEvidences[time_slice];
	}
	delete [] pEvidences;
}

void filtering(pnl::CDBN *pDBN)
{
	// number of time slices for unrolling
	const int numTimeSlices = 5;

	// let node 1 be always observed
	const int obsNodes[] = { 1 };  // 1st node ==> observed node
	pnl::valueVector obsNodesVals(1);

	// create evidence for every time-slice
	pnl::CEvidence **pEvidences = new pnl::CEvidence *[numTimeSlices];
	for (int time_slice = 0; time_slice < numTimeSlices; ++time_slice)
	{
		// generate random value
		// all nodes in the model are discrete
		obsNodesVals[0].SetInt(std::rand() % 2);
		pEvidences[time_slice] = pnl::CEvidence::Create(pDBN, 1, obsNodes, obsNodesVals);
	}

	// create an inference engine
	pnl::C1_5SliceJtreeInfEngine *pInfEng = pnl::C1_5SliceJtreeInfEngine::Create(pDBN);

	// create filtering procedure
	pInfEng->DefineProcedure(pnl::ptFiltering);

	std::cout << "========== Results of filtering " << std::endl;
	// for the prior network
	{
		const int queryPrior[] = { 0 };  // 0th node ==> hidden state
		const int queryPriorSize = 1;

		int time_slice = 0;
		pInfEng->EnterEvidence(&(pEvidences[time_slice]), 1);
		pInfEng->Filtering(time_slice);

		pInfEng->MarginalNodes(queryPrior, queryPriorSize);
		const pnl::CPotential *pQueryJPD = pInfEng->GetQueryJPD();

		std::cout << ">>> Query time-slice: " << time_slice << std::endl;
		pQueryJPD->Dump();
	}

	// for the transition network
	{
		const int query[] = { 0, 2 };  // 0th & 2nd nodes ==> hidden states
		const int querySize = 2;

		for (int time_slice = 1; time_slice < numTimeSlices; ++time_slice)
		{
			pInfEng->EnterEvidence(&(pEvidences[time_slice]), 1);
			pInfEng->Filtering(time_slice);

			pInfEng->MarginalNodes(query, querySize);
			const pnl::CPotential *pQueryJPD = pInfEng->GetQueryJPD();

			std::cout << ">>> Query time-slice: " << time_slice << std::endl;
			pQueryJPD->Dump();
		}
	}

	//
	delete pInfEng;
	for (int time_slice = 0; time_slice < numTimeSlices; ++time_slice)
	{
		delete pEvidences[time_slice];
	}
	delete [] pEvidences;
}

void fixed_lag_smoothing(pnl::CDBN *pDBN)
{
	// number of time slices for unrolling
	const int numTimeSlices = 5;

	// let node 1 be always observed
	const int obsNodes[] = { 1 };  // 1st node ==> observed node
	pnl::valueVector obsNodesVals(1);

	// create evidence for every time-slice
	pnl::CEvidence **pEvidences = new pnl::CEvidence *[numTimeSlices];
	for (int time_slice = 0; time_slice < numTimeSlices; ++time_slice)
	{
		// generate random value
		// all nodes in the model are discrete
		obsNodesVals[0].SetInt(std::rand() % 2);
		pEvidences[time_slice] = pnl::CEvidence::Create(pDBN, 1, obsNodes, obsNodesVals);
	}

	// create an inference engine
	pnl::C1_5SliceJtreeInfEngine *pInfEng = pnl::C1_5SliceJtreeInfEngine::Create(pDBN);

	// create fixed-lag smoothing (online)
	const int lag = 2;
	pInfEng->DefineProcedure(pnl::ptFixLagSmoothing, lag);

	std::cout << "========== Results of fixed-lag smoothing " << std::endl;
	// for the prior network
	{
		const int queryPrior[] = { 0 };  // 0th node ==> hidden state
		const int queryPriorSize = 1;

		int time_slice = 0;
		for ( ; time_slice < lag + 1; ++time_slice)
		{
			pInfEng->EnterEvidence(&(pEvidences[time_slice]), 1);
		}
		pInfEng->FixLagSmoothing(time_slice);

		pInfEng->MarginalNodes(queryPrior, queryPriorSize);
		const pnl::CPotential *pQueryJPD = pInfEng->GetQueryJPD();

		std::cout << ">>> Query time-slice: " << time_slice << std::endl;
		pQueryJPD->Dump();
	}

	// for the transition network
	{
		const int query[] = { 0, 2 };  // 0th & 2nd nodes ==> hidden states
		const int querySize = 2;

		for (int time_slice = lag + 1; time_slice < numTimeSlices; ++time_slice)
		{
			pInfEng->EnterEvidence(&(pEvidences[time_slice]), 1);
			pInfEng->FixLagSmoothing(time_slice);

			pInfEng->MarginalNodes(query, querySize);
			const pnl::CPotential *pQueryJPD = pInfEng->GetQueryJPD();

			std::cout << ">>> Query time-slice: " << time_slice << std::endl;
			pQueryJPD->Dump();
		}
	}

	//
	delete pInfEng;
	for (int time_slice = 0; time_slice < numTimeSlices; ++time_slice)
	{
		delete pEvidences[time_slice];
	}
	delete [] pEvidences;
}

// [ref] CompareViterbyArHMM() in "${PNL_ROOT}/c_pgmtk/tests/src/A1_5JTreeInfDBNCondGauss.cpp"
void maximum_probability_explanation(pnl::CDBN *pDBN)
{
	// number of time slices for unrolling
	const int numTimeSlices = 5;

	// let node 1 be always observed
	const pnl::intVector obsNodes(1, 1);  // 1st node ==> observed node
	pnl::valueVector obsNodesVals(1);

	// create values for evidence for every time-slice from t=0 to t=nTimeSlice
	pnl::pEvidencesVector evidences(numTimeSlices);
	for (int time_slice = 0; time_slice < numTimeSlices; ++time_slice)
	{
		//obsNodesVals[0].SetFlt(float(std::rand() % 10));  // compile-time error
		obsNodesVals[0].SetInt(std::rand() % 2);
		evidences[time_slice] = pnl::CEvidence::Create(pDBN, obsNodes, obsNodesVals);
	}

	// create an inference engine
	pnl::C1_5SliceJtreeInfEngine *pInfEng = pnl::C1_5SliceJtreeInfEngine::Create(pDBN);

	// create inference (smoothing) for DBN
	pInfEng->DefineProcedure(ptViterbi, numTimeSlices);
	pInfEng->EnterEvidence(&evidences.front(), numTimeSlices);
	pInfEng->FindMPE();

	pnl::intVector queryPrior(1), query(2);
	queryPrior[0] = 0;  // 0th node ==> hidden state
	query[0] = 0;  query[1] = 2;  // 0th & 2nd nodes ==> hidden states

	for (int time_slice = 0; time_slice < numTimeSlices; ++time_slice)
	{
		if (time_slice)  // for the transition network
		{
			pInfEng->MarginalNodes(&query.front(), query.size(), time_slice);
		}
		else  // for the prior network
		{
			pInfEng->MarginalNodes(&queryPrior.front(), queryPrior.size(), time_slice);
		}

		const pnl::CPotential *pQueryMPE = pInfEng->GetQueryMPE();

		std::cout << ">>> Query time-slice: " << time_slice << std::endl;

		int numNodes = 0;
		const int *domain = NULL;
		pQueryMPE->GetDomain(&numNodes, &domain);

		std::cout << " domain: ";
		for (int i = 0; i < numNodes; ++i)
		{
			std::cout << domain[i] << " ";
		}
		std::cout << std::endl;

		// TODO [check] >> is this code really correct?
		const pnl::CEvidence *mpeEvid = pInfEng->GetMPE();
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
}

}  // namespace local
}  // unnamed namespace

void dbn_example()
{
#if 1
	boost::scoped_ptr<pnl::CDBN> arHMM(pnl::CDBN::Create(pnl::pnlExCreateRndArHMM()));
#else
	boost::scoped_ptr<pnl::CDBN> arHMM(local::create_model());
#endif

	if (!arHMM)
	{
		std::cout << "can't create a probabilistic graphical model" << std::endl;
		return;
	}

	// get content of Graph
	arHMM->GetGraph()->Dump();
	pnl::CGraph *g = arHMM->GetGraph();

	//local::smoothing(arHMM.get());
	//local::filtering(arHMM.get());
	//local::fixed_lag_smoothing(arHMM.get());
	local::maximum_probability_explanation(arHMM.get());  // MPE by Viterbi algorithm
}
