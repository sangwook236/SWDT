#include "stdafx.h"
#include <pnl_dll.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>


namespace {
namespace local {

pnl::CDBN * create_ar_hmm()
{
/*
	an HMM with autoregressive Gaussian observations

		0 -> 2
		|    | 
		v    v
		1 -> 3 

	where
		0, 1, 2, 3 - tabular nodes (bivariate)
*/

	// for the transition network, nodes in the query with the number from 0 to N-1 belong to the time-slice ts-1
	// and nodes with the number from N to 2N-1 belong to the time-slice ts.
	// for the prior time-slice, for example, ts=0, nodes in the query must have numbers from 0 to N-1.
	// [ref] "Probabilistic Network Library: User Guide and Reference Manual", pp. 2-8 & p. 3-258
	//
	// for the case shown below
	//	-. N = 2
	//	-. for the prior network
	//		nodes belonging to the prior time-slice, ts=0 ==> 0 & 1
	//	-. for the transition network
	//		nodes belonging to the time-slice, ts-1 ==> 0 & 1
	//		nodes belonging to the time-slice, ts ==> 2 & 3

	// create static model
	const int numNodes = 4;  // number of nodes
	const int numNodeTypes = 1;  // number of node types (all nodes are discrete)

	// 1) first need to specify the graph structure of the model.
	const int numNeighbors[] = { 2, 2, 2, 2 };

	const int neigh0[] = { 1, 2 };
	const int neigh1[] = { 0, 3 };
	const int neigh2[] = { 0, 3 };
	const int neigh3[] = { 1, 2 };
	const int *neighs[] = { neigh0, neigh1, neigh2, neigh3 };

	const pnl::ENeighborType neighType0[] = { pnl::ntChild, pnl::ntChild };
	const pnl::ENeighborType neighType1[] = { pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType neighType2[] = { pnl::ntParent, pnl::ntChild };
	const pnl::ENeighborType neighType3[] = { pnl::ntParent, pnl::ntParent };
	const pnl::ENeighborType *neighTypes[] = { neighType0, neighType1, neighType2, neighType3 };
	
	pnl::CGraph *graph = pnl::CGraph::Create(numNodes, numNeighbors, neighs, neighTypes);
	
	// 2) create the Model Domain.
	const pnl::nodeTypeVector nodeTypes(numNodeTypes, pnl::CNodeType(true, 2));
	const pnl::intVector nodeAssociation(numNodes, 0);
	
	pnl::CModelDomain *modelDomain = pnl::CModelDomain::Create(nodeTypes, nodeAssociation);
	
	// 3) create static BNet with random matrices.
	pnl::CBNet *bnet = pnl::CBNet::CreateWithRandomMatrices(graph, modelDomain);
	
	// 4) create DBN.
	return pnl::CDBN::Create(bnet);
}

void smoothing(const boost::scoped_ptr<pnl::CDBN> &dbn)
{
	// number of time slices for unrolling
	const int numTimeSlices = 5;

	// let node 1 be always observed
	const int obsNodes[] = { 1 };  // 1st node ==> observed node
	pnl::valueVector obsNodesVals(1);

	// create evidence for every time-slice
	pnl::pEvidencesVector evidences(numTimeSlices);
	for (int time_slice = 0; time_slice < numTimeSlices; ++time_slice)
	{
		// generate random value
		// all nodes in the model are discrete
		obsNodesVals[0].SetInt(std::rand() % 2);
		evidences[time_slice] = pnl::CEvidence::Create(dbn.get(), 1, obsNodes, obsNodesVals);
	}

	// create an inference engine
	const boost::scoped_ptr<pnl::C1_5SliceJtreeInfEngine> infEngine(pnl::C1_5SliceJtreeInfEngine::Create(dbn.get()));

	// create smoothing procedure
	infEngine->DefineProcedure(pnl::ptSmoothing, numTimeSlices);
	// enter created evidences
	infEngine->EnterEvidence(&evidences.front(), numTimeSlices);
	// start smoothing process
	infEngine->Smoothing();

	// inference results gaining and representation
	std::cout << "========== results of smoothing " << std::endl;
	
	// choose query set of nodes for every time-slice
	// for the prior network
	{
		const int queryPrior[] = { 0 };  // 0th node ==> hidden state
		const int queryPriorSize = 1;

		// for the prior time-slice, ts = 0
		int time_slice = 0;
		// calculate joint probability distribution (JPD) & most probability explanation (MPE) (???)
		infEngine->MarginalNodes(queryPrior, queryPriorSize, time_slice);

		const pnl::CPotential *queryJPD = infEngine->GetQueryJPD();

		std::cout << ">>> query time-slice: " << time_slice << std::endl;

		int numNodes = 0;
		const int *domain = NULL;
		queryJPD->GetDomain(&numNodes, &domain);

		std::cout << " domain: ";
		for (int i = 0; i < numNodes; ++i)
		{
			std::cout << domain[i] << " ";
		}
		std::cout << std::endl;

		pnl::CMatrix<float> *jpdMat = queryJPD->GetMatrix(pnl::matTable);
		const pnl::EMatrixClass &type = jpdMat->GetMatrixClass();
		if (!(pnl::mcDense == type || pnl::mcNumericDense == type || pnl::mc2DNumericDense == type))
		{
			assert(0);
		}

		// graphical model has been created using dense matrix
		int numData = 0;
		const float *data = NULL;
		dynamic_cast<pnl::CDenseMatrix<float> *>(jpdMat)->GetRawData(&numData, &data);
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
			infEngine->MarginalNodes(query, querySize, time_slice);

			const pnl::CPotential *queryJPD = infEngine->GetQueryJPD();

			std::cout << ">>> query time-slice: " << time_slice << std::endl;
			// representation information using Dump()
			queryJPD->Dump();
		}
	}

	//
	for (int time_slice = 0; time_slice < numTimeSlices; ++time_slice)
	{
		delete evidences[time_slice];
	}
}

void filtering(const boost::scoped_ptr<pnl::CDBN> &dbn)
{
	// number of time slices for unrolling
	const int numTimeSlices = 5;

	// let node 1 be always observed
	const int obsNodes[] = { 1 };  // 1st node ==> observed node
	pnl::valueVector obsNodesVals(1);

	// create evidence for every time-slice
	pnl::pEvidencesVector evidences(numTimeSlices);
	for (int time_slice = 0; time_slice < numTimeSlices; ++time_slice)
	{
		// generate random value
		// all nodes in the model are discrete
		obsNodesVals[0].SetInt(std::rand() % 2);
		evidences[time_slice] = pnl::CEvidence::Create(dbn.get(), 1, obsNodes, obsNodesVals);
	}

	// create an inference engine
	const boost::scoped_ptr<pnl::C1_5SliceJtreeInfEngine> infEngine(pnl::C1_5SliceJtreeInfEngine::Create(dbn.get()));

	// create filtering procedure
	infEngine->DefineProcedure(pnl::ptFiltering);

	std::cout << "========== results of filtering " << std::endl;
	// for the prior network
	{
		const int queryPrior[] = { 0 };  // 0th node ==> hidden state
		const int queryPriorSize = 1;

		int time_slice = 0;
		infEngine->EnterEvidence(&(evidences[time_slice]), 1);
		infEngine->Filtering(time_slice);

		infEngine->MarginalNodes(queryPrior, queryPriorSize);
		const pnl::CPotential *queryJPD = infEngine->GetQueryJPD();

		std::cout << ">>> query time-slice: " << time_slice << std::endl;
		queryJPD->Dump();
	}

	// for the transition network
	{
		const int query[] = { 0, 2 };  // 0th & 2nd nodes ==> hidden states
		const int querySize = 2;

		for (int time_slice = 1; time_slice < numTimeSlices; ++time_slice)
		{
			infEngine->EnterEvidence(&(evidences[time_slice]), 1);
			infEngine->Filtering(time_slice);

			infEngine->MarginalNodes(query, querySize);
			const pnl::CPotential *queryJPD = infEngine->GetQueryJPD();

			std::cout << ">>> query time-slice: " << time_slice << std::endl;
			queryJPD->Dump();
		}
	}

	//
	for (int time_slice = 0; time_slice < numTimeSlices; ++time_slice)
	{
		delete evidences[time_slice];
	}
}

void fixed_lag_smoothing(const boost::scoped_ptr<pnl::CDBN> &dbn)
{
	// number of time slices for unrolling
	const int numTimeSlices = 5;

	// let node 1 be always observed
	const int obsNodes[] = { 1 };  // 1st node ==> observed node
	pnl::valueVector obsNodesVals(1);

	// create evidence for every time-slice
	pnl::pEvidencesVector evidences(numTimeSlices);
	for (int time_slice = 0; time_slice < numTimeSlices; ++time_slice)
	{
		// generate random value
		// all nodes in the model are discrete
		obsNodesVals[0].SetInt(std::rand() % 2);
		evidences[time_slice] = pnl::CEvidence::Create(dbn.get(), 1, obsNodes, obsNodesVals);
	}

	// create an inference engine
	const boost::scoped_ptr<pnl::C1_5SliceJtreeInfEngine> infEngine(pnl::C1_5SliceJtreeInfEngine::Create(dbn.get()));

	// create fixed-lag smoothing (online)
	const int lag = 2;
	infEngine->DefineProcedure(pnl::ptFixLagSmoothing, lag);

	std::cout << "========== results of fixed-lag smoothing " << std::endl;
	// for the prior network
	{
		const int queryPrior[] = { 0 };  // 0th node ==> hidden state
		const int queryPriorSize = 1;

		int time_slice = 0;
		for ( ; time_slice < lag + 1; ++time_slice)
		{
			infEngine->EnterEvidence(&(evidences[time_slice]), 1);
		}
		infEngine->FixLagSmoothing(time_slice);

		infEngine->MarginalNodes(queryPrior, queryPriorSize);
		const pnl::CPotential *queryJPD = infEngine->GetQueryJPD();

		std::cout << ">>> query time-slice: " << time_slice << std::endl;
		queryJPD->Dump();
	}

	// for the transition network
	{
		const int query[] = { 0, 2 };  // 0th & 2nd nodes ==> hidden states
		const int querySize = 2;

		for (int time_slice = lag + 1; time_slice < numTimeSlices; ++time_slice)
		{
			infEngine->EnterEvidence(&(evidences[time_slice]), 1);
			infEngine->FixLagSmoothing(time_slice);

			infEngine->MarginalNodes(query, querySize);
			const pnl::CPotential *queryJPD = infEngine->GetQueryJPD();

			std::cout << ">>> query time-slice: " << time_slice << std::endl;
			queryJPD->Dump();
		}
	}

	//
	for (int time_slice = 0; time_slice < numTimeSlices; ++time_slice)
	{
		delete evidences[time_slice];
	}
}

// [ref] CompareViterbyArHMM() in "${PNL_ROOT}/c_pgmtk/tests/src/A1_5JTreeInfDBNCondGauss.cpp"
void maximum_probability_explanation(const boost::scoped_ptr<pnl::CDBN> &dbn)
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
		evidences[time_slice] = pnl::CEvidence::Create(dbn.get(), obsNodes, obsNodesVals);
	}

	// create an inference engine
	pnl::C1_5SliceJtreeInfEngine *infEngine = pnl::C1_5SliceJtreeInfEngine::Create(dbn.get());

	// create inference (smoothing) for DBN
	infEngine->DefineProcedure(ptViterbi, numTimeSlices);
	infEngine->EnterEvidence(&evidences.front(), numTimeSlices);
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

		const pnl::CPotential *pQueryMPE = infEngine->GetQueryMPE();

		std::cout << ">>> query time-slice: " << time_slice << std::endl;

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
		delete evidences[i];
	}
}

// [ref] "Probabilistic Network Library: User Guide and Reference Manual", pp. 2-32 ~ 33
void learn_dbn_with_ar_gaussian_observations(const boost::scoped_ptr<pnl::CDBN> &dbn)
{
	// the definitions of time series & time slices
	//	[ref] "Probabilistic Network Library: User Guide and Reference Manual", pp. 3-290 & 2-32
	const int numTimeSeries = 500;

	// define number of slices in the every time series
	pnl::intVector numSlices(numTimeSeries);
	pnl::pnlRand(numTimeSeries, &numSlices.front(), 3, 20);
	
	// generate evidences in a random way
	pnl::pEvidencesVecVector evidences;
	dbn->GenerateSamples(&evidences, numSlices);

	// create DBN for learning
	const boost::scoped_ptr<pnl::CDBN> dbnToLearn(pnl::CDBN::Create(pnl::pnlExCreateRndArHMM()));

	// create learning engine
	const boost::scoped_ptr<pnl::CEMLearningEngineDBN> learnEngine(pnl::CEMLearningEngineDBN::Create(dbnToLearn.get()));
	
	// set data for learning
	learnEngine->SetData(evidences);

	// start learning
	try
	{
		learnEngine->Learn();
	}
	catch (const pnl::CAlgorithmicException &e)
	{
		std::cout << "fail to learn parameters of a DBN" << e.GetMessage() << std::endl;
		return;
	}

	//
	for (int i = 0; i < evidences.size(); ++i)
	{
		for (int j = 0; j < evidences[i].size(); ++j)
			delete evidences[i][j];
	}

	//
	const int numFactors = dbnToLearn->GetNumberOfFactors();
	for (int i = 0; i < numFactors; ++i)
	{
		int numNodes;
		const int *domain;
		const pnl::CFactor *cpd = dbn->GetFactor(i);
		cpd->GetDomain(&numNodes, &domain);

		std::cout << " #-- node " << domain[numNodes - 1] << " has the parents ";
		for (int node = 0; node < numNodes - 1; ++node)
		{
			std::cout << domain[node] << " ";
		}
		std::cout << std::endl;

		std::cout << " #-- conditional probability distribution for node " << i << std::endl;

		std::cout << " #-- initial model" << std::endl;
		const pnl::CTabularDistribFun *distribFun = static_cast<const pnl::CTabularDistribFun *>(cpd->GetDistribFun());
		distribFun->Dump();

		std::cout << " #-- model after learning" << std::endl;
		cpd = dbnToLearn->GetFactor(i);
		distribFun = static_cast<const pnl::CTabularDistribFun *>(cpd->GetDistribFun());
		distribFun->Dump();
	}
}

}  // namespace local
}  // unnamed namespace

void dbn_example()
{
	// infer DBN with AR Gaussian observations
	std::cout << "========== infer DBN with AR Gaussian observations" << std::endl;
	{
#if 1
		const boost::scoped_ptr<pnl::CDBN> arHMM(pnl::CDBN::Create(pnl::pnlExCreateRndArHMM()));
		//const boost::scoped_ptr<pnl::CDBN> arHMM(pnl::CDBN::Create(pnl::pnlExCreateCondGaussArBNet()));
#else
		const boost::scoped_ptr<pnl::CDBN> arHMM(local::create_ar_hmm());
#endif

		if (!arHMM)
		{
			std::cout << "fail to create a probabilistic graphical model at " << __LINE__ << " in " << __FILE__ << std::endl;
			return;
		}

		// get content of Graph
		{
			arHMM->GetGraph()->Dump();
		}

		local::smoothing(arHMM);
		//local::filtering(arHMM);
		//local::fixed_lag_smoothing(arHMM);
		//local::maximum_probability_explanation(arHMM);  // MPE by Viterbi algorithm
	}

	// learn DBN with AR Gaussian observations
	std::cout << "\n========== learn DBN with AR Gaussian observations" << std::endl;
	{
		const boost::scoped_ptr<pnl::CDBN> arHMM(pnl::CDBN::Create(pnl::pnlExCreateRndArHMM()));
		//const boost::scoped_ptr<pnl::CDBN> arHMM(pnl::CDBN::Create(pnl::pnlExCreateCondGaussArBNet()));

		if (!arHMM)
		{
			std::cout << "fail to create a probabilistic graphical model at " << __LINE__ << " in " << __FILE__ << std::endl;
			return;
		}

		local::learn_dbn_with_ar_gaussian_observations(arHMM);
	}
}
