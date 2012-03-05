//#include "stdafx.h"
#include <pnl_dll.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>


namespace {
namespace local {

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/APearlInfEngineMRF2.cpp
pnl::CMRF2 * create_simple_mrf2_model()
{
/*
	the model is

	                0
				  /   \
				 /     \
                /       \
			   1         4
			  / \       / \
             /   \     /   \
			2     3   5     6
                           /|\
						  / | \
						 7  8  9
                               |
                               |
                               10

	where
		0, 1, 2, ..., 10 - bivariate tabular nodes
*/

	const int numNodes = 11; //4;//7;
	const int numNodeTypes = 1;
	const int numCliques = 10; //3;// 6;

	const pnl::nodeTypeVector nodeTypes(numNodeTypes, pnl::CNodeType(true, 2));  // discrete & binary
	const pnl::intVector nodeAssociation(numNodes, 0);

	//const boost::scoped_ptr<pnl::CModelDomain> modelDomain(pnl::CModelDomain::Create(nodeTypes, nodeAssociation));  // run-time error
	pnl::CModelDomain *modelDomain = pnl::CModelDomain::Create(nodeTypes, nodeAssociation);

	// create graphical model by cliques;
	const std::vector<int> cliqueSizes(numCliques, 2);

	const int clique0[] = { 0, 1 };
	const int clique1[] = { 1, 2 };
	const int clique2[] = { 1, 3 };
	const int clique3[] = { 0, 4 };
	const int clique4[] = { 4, 5 };
	const int clique5[] = { 4, 6 };
	const int clique6[] = { 6, 7 };
	const int clique7[] = { 6, 8 };
	const int clique8[] = { 6, 9 };
	const int clique9[] = { 9, 10 };
	const int *cliques[] = { clique0, clique1, clique2, clique3, clique4, clique5, clique6, clique7, clique8, clique9 };

	pnl::CMRF2 *mrf2 = pnl::CMRF2::Create(numCliques, &cliqueSizes.front(), cliques, modelDomain);

	mrf2->GetGraph()->Dump();

	// create container for factors
	mrf2->AllocFactors();

	// to create factors we need to create their tables
	// create array of data for every parameter
	const float table0[] = { 0.6f, 0.4f, 0.8f, 0.2f };  // 2 x 2
	const float table1[] = { 0.5f, 0.5f, 0.7f, 0.3f };  // 2 x 2
	const float table2[] = { 0.1f, 0.9f, 0.3f, 0.7f };  // 2 x 2
	const float table3[] = { 0.1f, 0.9f, 0.2f, 0.8f }; //{ 0.01f, 0.99f, 0.02f, 0.98f };  // 2 x 2
	const float table4[] = { 0.2f, 0.8f, 0.3f, 0.7f };  // 2 x 2
	const float table5[] = { 0.4f, 0.6f, 0.6f, 0.4f };  // 2 x 2
	const float table6[] = { 0.8f, 0.2f, 0.9f, 0.1f };  // 2 x 2
	const float table7[] = { 1.0f, 0.0f, 0.0f, 1.0f };  // 2 x 2
	const float table8[] = { 0.5f, 0.5f, 0.5f, 0.5f };  // 2 x 2
	const float table9[] = { 0.1f, 0.9f, 0.2f, 0.8f };  // 2 x 2
	const float *tables[] = { table0, table1, table2, table3, table4, table5, table6, table7, table8, table9 };

	// number of factors is the same as number of cliques - one per clique
	std::vector<pnl::CFactor *> potentials(numCliques, (pnl::CFactor *)NULL);
	for (int i = 0; i < numCliques; ++i)
	{
		potentials[i] = pnl::CTabularPotential::Create(cliques[i], 2, modelDomain);
		potentials[i]->AllocMatrix(tables[i], pnl::matTable);

		mrf2->AttachFactor((pnl::CTabularPotential *)potentials[i]);
	}

	return mrf2;
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/APearlInfEngineMRF2.cpp
void infer_simple_mrf2_using_belief_propagation_algorithm(const boost::scoped_ptr<pnl::CMRF2> &mrf2)
{
	const int numNodes = mrf2->GetNumberOfNodes();  // TODO [check] >> is it correct?

	// create evidence
	const boost::scoped_ptr<pnl::CEvidence> emptyEvid(pnl::CEvidence::Create(mrf2.get(), 0, NULL, pnl::valueVector()));

	// now we can compare naive inf. results with Pearl inf. results
	// belief propagation (Pearl inference)
	const boost::scoped_ptr<pnl::CPearlInfEngine> pearlInfEngine(pnl::CPearlInfEngine::Create(mrf2.get()));
	pearlInfEngine->EnterEvidence(emptyEvid.get());

	const boost::scoped_ptr<pnl::CNaiveInfEngine> naiveInfEngine(pnl::CNaiveInfEngine::Create(mrf2.get()));
	naiveInfEngine->EnterEvidence(emptyEvid.get());

/*
	const boost::scoped_ptr<pnl::CJtreeInfEngine> juncTreeInfEngine(pnl::CJtreeInfEngine::Create(mrf2.get()));
	juncTreeInfEngine->EnterEvidence(emptyEvid.get());
*/

	// inference without evidence
	{
/*
		// data from Matlab
		const float marginal0[] = { 0.5f, 0.5f };
		const float marginal1[] = { 0.7f, 0.3f };
		const float marginal2[] = { 0.56f, 0.44f };
		const float marginal3[] = { 0.16f, 0.84f }; //{ 0.16f, 0.84f };
		const float marginal4[] = { 0.15f, 0.85f }; //{ 0.015f, 0.985f };
		const float marginal5[] = { 0.285f, 0.715f }; //{ 0.2985f, 0.7015f };
		const float marginal6[] = { 0.57f, 0.43f }; //{ 0.597f, 0.403f };
		const float marginal7[] = { 0.843f, 0.157f }; //{ 0.8403f, 0.1597f };
		const float marginal8[] = { 0.57f, 0.43f }; //{ 0.597f, 0.403f };
		const float marginal9[] = { 0.5f, 0.5f };
		const float marginal10[] = { 0.15f, 0.85f };
		const float *marginals[] = {
			marginal0, marginal1, marginal2, marginal3,
			marginal4, marginal5, marginal6, marginal7,
			marginal8, marginal9, marginal10
		};
*/

		const int querySize = 1;
		const float eps = 1e-5f;

		std::vector<int> query(querySize);
		for (int i = 0; i < numNodes; ++i)
		{
			query[0] = i;
			pearlInfEngine->MarginalNodes(&query.front(), querySize);
			naiveInfEngine->MarginalNodes(&query.front(), querySize);
			//juncTreeInfEngine->MarginalNodes(&query.front(), querySize);

			const pnl::CPotential *jpdForPearl = pearlInfEngine->GetQueryJPD();
			const pnl::CPotential *jpdForNaive = naiveInfEngine->GetQueryJPD();
			//const pnl::CPotential *jpdForJTree = juncTreeInfEngine->GetQueryJPD();

			if (!jpdForPearl->IsFactorsDistribFunEqual(jpdForNaive, eps, 0))
			{
				// TODO [implement] >>
				std::cout << "results of Pearl inference & naive inference are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
		}
	}

	//
	{
		// we can add evidence and compare results with results from naive infrence engine
		const int numObsNodes = 1;

		std::vector<int> obsNodes(numObsNodes);
		pnl::valueVector obsValues(numObsNodes, pnl::Value(0));
		pnl::intVector residuaryNodesFor ;
		for (int i = 0; i < numNodes; ++i)
		{
			residuaryNodesFor.push_back(i);
		}
		for (int i = 0; i < numObsNodes; ++i)
		{
			const int j = std::rand() % (numNodes - i);
			obsNodes[i] = residuaryNodesFor[j];
			residuaryNodesFor.erase(residuaryNodesFor.begin() + j);
			obsValues[i].SetInt(std::rand() % (mrf2->GetNodeType(obsNodes[i])->GetNodeSize()));
		}
		residuaryNodesFor.clear();

		const boost::scoped_ptr<pnl::CEvidence> evidWithObs(pnl::CEvidence::Create(mrf2.get(), numObsNodes, &obsNodes.front(), obsValues));

		pearlInfEngine->EnterEvidence(evidWithObs.get());
		//const int numIters = pearlInfEngine->GetNumberOfProvideIterations();
		naiveInfEngine->EnterEvidence(evidWithObs.get());

		const int querySize = 1;
		const float eps = 1e-5f;

		std::vector<int> query(querySize);
		for (int i = 0; i < numNodes; ++i)
		{
			query[0] = i;
			pearlInfEngine->MarginalNodes(&query.front(), querySize);
			naiveInfEngine->MarginalNodes(&query.front(), querySize);

			const pnl::CPotential *jpdForPearl = pearlInfEngine->GetQueryJPD();
			const pnl::CPotential *jpdForNaive = naiveInfEngine->GetQueryJPD();

			if (!jpdForPearl->IsFactorsDistribFunEqual(jpdForNaive, eps, 0))
			{
				// TODO [implement] >>
				std::cout << "results of Pearl inference & naive inference are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
		}
	}
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/AGaussianMRF2.cpp
pnl::CMRF2 * create_gaussian_mrf2_1()
{
/*
	the model is

	             0 -- 1 -- 2

	where
		0, 1, 2 - univariate Gaussian nodes
*/

	const int numNodes = 3;
	const int numNodeTypes = 3;
	const int numCliques = 2;

	const pnl::nodeTypeVector nodeTypes(numNodeTypes, pnl::CNodeType(false, 1));
	const pnl::intVector nodeAssociation(numNodes, 0);  // { 0, 0, 0 }

/*
	const int cliqueSizes[] = { 2, 2 };

	const int clique0[2] = { 0, 1 };
	const int clique1[2] = { 1, 2 };
	const int *cliques[] = { clique0, clique1 };
*/
	pnl::intVecVector cliques(numCliques);
	pnl::intVector clique0;  // { 0, 1 }
	clique0.push_back(0);
	clique0.push_back(1);
	cliques[0] = clique0;
	pnl::intVector clique1;  // { 1, 2 }
	clique1.push_back(1);
	clique1.push_back(2);
	cliques[1] = clique1;

	pnl::CMRF2 *mrf2 = pnl::CMRF2::Create(numNodes, nodeTypes, nodeAssociation, cliques);

	//
	const float mean0[] = { 1.0f, 2.0f };  // 2 <- 1 + 1
	const float mean1[] = { 4.0f, 3.0f };  // 2 <- 1 + 1
	const float *means[] = { mean0, mean1 };

	const float cov0[] = { 3.0f, 3.0f, 3.0f, 4.0f };  // 2 x 2
	const float cov1[] = { 1.0f, 1.0f, 1.0f, 3.0f };  // 2 x 2
	const float *covs[] = { cov0, cov1 };

	const float coeffs[] = { 1.0f, 1.0f };

	mrf2->AllocFactors();

	pnl::CModelDomain *modelDomain = mrf2->GetModelDomain();
	pnl::CFactorGraph *factorGraph = pnl::CFactorGraph::Create(modelDomain, numCliques);

	std::vector<pnl::CFactor *> potentials(numCliques, (pnl::CFactor *)NULL);
	//const pnl::CNodeType *domainNodeTypes[2];
	for (int i = 0; i < numCliques; ++i)
	{
/*
		for (int j = 0; j < 2; ++j)
		{
			domainNodeTypes[j] = mrf2->GetNodeType(cliques[i][j]);
		}
*/
		potentials[i] = pnl::CGaussianPotential::Create(&cliques[i].front(), 2, modelDomain);

		static_cast<pnl::CGaussianPotential *>(potentials[i])->SetCoefficient(coeffs[i], 1);
		potentials[i]->AllocMatrix(means[i], pnl::matMean);
		potentials[i]->AllocMatrix(covs[i], pnl::matCovariance);
		mrf2->AttachFactor(dynamic_cast<pnl::CGaussianPotential *>(potentials[i]));

		// TODO [check] >> i think this implementation does not need
/*
		factorGraph->AllocFactor(2, &cliques[i].front());

		pnl::pFactorVector factors;
		factorGraph->GetFactors(2, &cliques[i].front(), &factors);
		static_cast<pnl::CGaussianPotential *>(factors[0])->SetCoefficient(coeffs[i], 1);
		factors[0]->AllocMatrix(means[i], pnl::matMean);
		factors[0]->AllocMatrix(covs[i], pnl::matCovariance);
*/
	}

	return mrf2;
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/AGaussianMRF2.cpp
void infer_gaussian_mrf2_using_inference_algorithm_1(const boost::scoped_ptr<pnl::CMRF2> &mrf2)
{
	const int numNodes = mrf2->GetNumberOfNodes();  // TODO [check] >> is it correct?

	//
	{
		const boost::scoped_ptr<pnl::CEvidence> emptyEvid(pnl::CEvidence::Create(mrf2.get(), 0, NULL, pnl::valueVector()));

		const boost::scoped_ptr<pnl::CNaiveInfEngine> emptyNaiveInfEngine(pnl::CNaiveInfEngine::Create(mrf2.get()));
		emptyNaiveInfEngine->EnterEvidence(emptyEvid.get());

		const boost::scoped_ptr<pnl::CFactorGraph> factorGraph(pnl::CFactorGraph::ConvertFromMNet(mrf2.get()));  // TODO [check] >> is it correct?
		// belief propagation on a factor graph model
		const boost::scoped_ptr<pnl::CFGSumMaxInfEngine> emptyFGSumMaxInfEngine(pnl::CFGSumMaxInfEngine::Create(factorGraph.get()));
		emptyFGSumMaxInfEngine->EnterEvidence(emptyEvid.get());

		const boost::scoped_ptr<pnl::CInfEngine> emptyJTreeInfEngine(pnl::CJtreeInfEngine::Create(mrf2.get()));
		emptyJTreeInfEngine->EnterEvidence(emptyEvid.get());

		// belief propagation (Pearl inference)
		const boost::scoped_ptr<pnl::CPearlInfEngine> emptyPearlInfEngine(pnl::CPearlInfEngine::Create(mrf2.get()));
		emptyPearlInfEngine->EnterEvidence(emptyEvid.get());

		int query = 0;
		const float eps = 1e-4f;
		for (int i = 0; i < numNodes; ++i)
		{
			query = i;
			emptyNaiveInfEngine->MarginalNodes(&query, 1);
			const pnl::CPotential *jpdForNaive = emptyNaiveInfEngine->GetQueryJPD();

			emptyPearlInfEngine->MarginalNodes(&query, 1);
			const pnl::CPotential *jpdForPearl = emptyPearlInfEngine->GetQueryJPD();

			emptyJTreeInfEngine->MarginalNodes(&query, 1);
			const pnl::CPotential *jpdForJTree = emptyJTreeInfEngine->GetQueryJPD();

			emptyFGSumMaxInfEngine->MarginalNodes(&query, 1);
			const pnl::CPotential *jpdForFGSumMax = emptyFGSumMaxInfEngine->GetQueryJPD();

			if (!jpdForPearl->IsFactorsDistribFunEqual(jpdForNaive, eps, 0))
			{
				// TODO [implement] >>
				std::cout << "results of Pearl inference & naive inference are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
			if (!jpdForFGSumMax->IsFactorsDistribFunEqual(jpdForPearl, eps, 0))
			{
				// TODO [implement] >>
				std::cout << "results of sum-max inference & Pearl inference are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
			if (!jpdForJTree->IsFactorsDistribFunEqual(jpdForNaive, eps, 0))
			{
				// TODO [implement] >>
				std::cout << "results of junction tree inference & naive inference are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
			if (!jpdForJTree->IsFactorsDistribFunEqual(jpdForPearl, eps, 0))
			{
				// TODO [implement] >>
				std::cout << "results of junction tree inference & Pearl inference are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
		}
	}

	//
	{
		const int obsNode = 1;

		pnl::valueVector obsVal(1, pnl::Value(0));
		obsVal[0].SetFlt(3.0f);

		const boost::scoped_ptr<pnl::CEvidence> oneEvid(pnl::CEvidence::Create(mrf2.get(), 1, &obsNode, obsVal));

		const boost::scoped_ptr<pnl::CNaiveInfEngine> oneNaiveInfEngine(pnl::CNaiveInfEngine::Create(mrf2.get()));
		oneNaiveInfEngine->EnterEvidence(oneEvid.get());

		// belief propagation (Pearl inference)
		const boost::scoped_ptr<pnl::CPearlInfEngine> onePearlInfEngine(pnl::CPearlInfEngine::Create(mrf2.get()));
		onePearlInfEngine->SetMaxNumberOfIterations(10);
		onePearlInfEngine->EnterEvidence(oneEvid.get());

		const boost::scoped_ptr<pnl::CFactorGraph> factorGraph(pnl::CFactorGraph::ConvertFromMNet(mrf2.get()));  // TODO [check] >> is it correct?
		// belief propagation on a factor graph model
		const boost::scoped_ptr<pnl::CFGSumMaxInfEngine> oneFGSumMaxInfEngine(pnl::CFGSumMaxInfEngine::Create(factorGraph.get()));
		oneFGSumMaxInfEngine->EnterEvidence(oneEvid.get());

		const boost::scoped_ptr<pnl::CJtreeInfEngine> oneJTreeInfEngine(pnl::CJtreeInfEngine::Create(mrf2.get()));
		oneJTreeInfEngine->EnterEvidence(oneEvid.get());

		int query = 0;
		const float eps = 1e-4f;
		for (int i = 0; i < numNodes; ++i)
		{
			query = i;
			oneNaiveInfEngine->MarginalNodes(&query, 1);
			const pnl::CPotential *jpdForNaive = oneNaiveInfEngine->GetQueryJPD();

			onePearlInfEngine->MarginalNodes(&query, 1);
			const pnl::CPotential *jpdForPearl = onePearlInfEngine->GetQueryJPD();

			oneJTreeInfEngine->MarginalNodes(&query, 1);
			const pnl::CPotential *jpdForJTree = oneJTreeInfEngine->GetQueryJPD();

			oneFGSumMaxInfEngine->MarginalNodes(&query, 1);
			const pnl::CPotential *jpdForFGSumMax = oneFGSumMaxInfEngine->GetQueryJPD();

			if (!jpdForPearl->IsFactorsDistribFunEqual(jpdForNaive, eps, 0))
			{
				// TODO [implement] >>
				std::cout << "results of Pearl inference & naive inference are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
			if (!jpdForFGSumMax->IsFactorsDistribFunEqual(jpdForPearl, eps, 0))
			{
				// TODO [implement] >>
				std::cout << "results of sum-max inference & Pearl inference are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
			if (!jpdForFGSumMax->IsFactorsDistribFunEqual(jpdForNaive, eps, 0))
			{
				// TODO [implement] >>
				std::cout << "results of sum-max inference & naive inference are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
			if (!jpdForJTree->IsFactorsDistribFunEqual(jpdForNaive, eps, 0))
			{
				// TODO [implement] >>
				std::cout << "results of junction tree inference & naive inference are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
		}
	}
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/AGaussianMRF2.cpp
pnl::CMRF2 * create_gaussian_mrf2_2()
{
/*
	the model is

	             0 -- 1 -- 2

	where
		0 - bivariate Gaussian node
		1 - trivariate Gaussian node
		2 - univariate Gaussian node
*/

	const int numNodes = 3;
	const int numNodeTypes = 3;
	const int numCliques = 2;

	pnl::nodeTypeVector nodeTypes(numNodeTypes);
	for (int i = 0; i < numNodeTypes; ++i)
	{
		nodeTypes[i].SetType(false, i + 1);
	}

	pnl::intVector nodeAssociation(numNodes, 0);  // { 1, 2, 0 }
	nodeAssociation[0] = 1;
	nodeAssociation[1] = 2;
	nodeAssociation[2] = 0;

/*
	const int cliqueSizes[] = { 2, 2 };

	const int clique0[2] = { 0, 1 };
	const int clique1[2] = { 1, 2 };
	const int *cliques[] = { clique0, clique1 };
*/
	pnl::intVecVector cliques(numCliques);
	pnl::intVector clique0;  // { 0, 1 }
	clique0.push_back(0);
	clique0.push_back(1);
	cliques[0] = clique0;
	pnl::intVector clique1;  // { 1, 2 }
	clique1.push_back(1);
	clique1.push_back(2);
	cliques[1] = clique1;

	pnl::CMRF2 *mrf2 = pnl::CMRF2::Create(numNodes, nodeTypes, nodeAssociation, cliques);

	//
	const float mean0[] = { 0.6f, 0.4f, 1.3f, 1.7f, 1.9f };  // 5 <- 2 + 3
	const float mean1[] = { 1.6f, 1.7f, 1.8f, 2.1f };  // 4 <- 3 + 1
	const float *means[] = { mean0, mean1 };

	const float cov0[] = { 7.4f, 7.5f, 7.6f, 7.4f, 7.3f, 7.5f, 7.2f, 7.3f, 7.3f, 7.5f, 7.6f, 7.3f, 7.8f, 7.1f, 7.1f, 7.4f, 7.3f, 7.1f, 7.1f, 7.6f, 7.3f, 7.5f, 7.1f, 7.6f, 7.3f  };  // 5 x 5
	//const float cov0[] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0, 1.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.3f  };  // 5 x 5
	const float cov1[] = { 3.0f, 4.0f, 5.0f, 6.0f, 4.0f, 8.0f, 9.0f, 1.0f, 5.0f, 9.0f, 3.0f, 4.0f, 6.0f, 1.0f, 4.0f, 8.0f };  // 4 x 4
	//const float cov1[] = { 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f };  // 4 x 4
	const float *covs[] = { cov0, cov1 };

	const float coeffs[] = { 1.0f, 1.0f };

	mrf2->AllocFactors();

	pnl::CModelDomain *modelDomain = mrf2->GetModelDomain();
	std::vector<pnl::CFactor *> potentials(numCliques, (pnl::CFactor *)NULL);
	//const pnl::CNodeType *domainNodeTypes[2];
	//const boost::scoped_ptr<pnl::CFactors> factors(pnl::CFactors::Create(numCliques));  // TODO [check] >>
	for (int i = 0; i < numCliques; ++i)
	{
/*
		for (int j = 0; j < 2; ++j)
		{
			domainNodeTypes[j] = mrf2->GetNodeType(cliques[i][j]);
		}
*/

		potentials[i] = pnl::CGaussianPotential::Create(&cliques[i].front(), 2, modelDomain);
		static_cast<pnl::CGaussianPotential *>(potentials[i])->SetCoefficient(coeffs[i], 0);
		potentials[i]->AllocMatrix(means[i], pnl::matMean);
		potentials[i]->AllocMatrix(covs[i], pnl::matCovariance);
		mrf2->AttachFactor((pnl::CGaussianPotential *)potentials[i]);

		// TODO [check] >> i think this implementation does not need
		//factors->AddFactor(potentials[i]->Clone());
	}

	return mrf2;
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/AGaussianMRF2.cpp
void infer_gaussian_mrf2_using_inference_algorithm_2(const boost::scoped_ptr<pnl::CMRF2> &mrf2)
{
	const int numNodes = mrf2->GetNumberOfNodes();  // TODO [check] >> is it correct?

	// TODO [check] >> is it correct?
	const boost::scoped_ptr<pnl::CFactorGraph> factorGraph(pnl::CFactorGraph::ConvertFromMNet(mrf2.get()));
	// create the factor graph based on factors
	//const boost::scoped_ptr<pnl::CFactorGraph> factorGraph = pnl::CFactorGraph::Create(mrf2->GetModelDomain(), factors.get()));

	//
	{
		const boost::scoped_ptr<pnl::CEvidence> emptyEvid(pnl::CEvidence::Create(mrf2.get(), 0, NULL, pnl::valueVector()));

		// belief propagation on a factor graph model
		const boost::scoped_ptr<pnl::CFGSumMaxInfEngine> emptyFGSumMaxInfEngine(pnl::CFGSumMaxInfEngine::Create(factorGraph.get()));
		emptyFGSumMaxInfEngine->EnterEvidence(emptyEvid.get());

		const boost::scoped_ptr<pnl::CInfEngine> emptyNaiveInfEngine(pnl::CNaiveInfEngine::Create(mrf2.get()));
		emptyNaiveInfEngine->EnterEvidence(emptyEvid.get());

		// belief propagation (Pearl inference)
		const boost::scoped_ptr<pnl::CInfEngine> emptyPearlInfEngine(pnl::CPearlInfEngine::Create(mrf2.get()));
		emptyPearlInfEngine->EnterEvidence(emptyEvid.get());

		int query = 0;
		const float eps = 1e-5f;
		for (int i = 0; i < numNodes; ++i)
		{
			query = i;
			emptyFGSumMaxInfEngine->MarginalNodes(&query, 1);
			const pnl::CPotential *jpdForFGSumMax = emptyFGSumMaxInfEngine->GetQueryJPD();
			emptyNaiveInfEngine->MarginalNodes(&query, 1);
			const pnl::CPotential *jpdForNaive = emptyNaiveInfEngine->GetQueryJPD();
			emptyPearlInfEngine->MarginalNodes(&query, 1);
			const pnl::CPotential *jpdForPearl = emptyPearlInfEngine->GetQueryJPD();

			if (!jpdForFGSumMax->IsFactorsDistribFunEqual(jpdForNaive, eps, 0))
			{
				// TODO [implement] >>
				std::cout << "results of sum-max inference & naive inference are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
			if (!jpdForPearl->IsFactorsDistribFunEqual(jpdForNaive, eps, 0))
			{
				// TODO [implement] >>
				std::cout << "results of Pearl inference & naive inference are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
		}
	}

	//
	{
		const int obsNode = 1;

		pnl::valueVector vals(3, pnl::Value(0));
		vals[0].SetFlt(1.0f);
		vals[1].SetFlt(2.0f);
		vals[2].SetFlt(3.0f);

		const boost::scoped_ptr<pnl::CEvidence> oneEvid(pnl::CEvidence::Create(mrf2.get(), 1, &obsNode, vals));

		// belief propagation on a factor graph model
		const boost::scoped_ptr<pnl::CFGSumMaxInfEngine> oneFGSumMaxInfEngine(pnl::CFGSumMaxInfEngine::Create(factorGraph.get()));
		oneFGSumMaxInfEngine->EnterEvidence(oneEvid.get());

		const boost::scoped_ptr<pnl::CInfEngine> oneNaiveInfEngine(pnl::CNaiveInfEngine::Create(mrf2.get()));
		oneNaiveInfEngine->EnterEvidence(oneEvid.get());

		// belief propagation (Pearl inference)
		const boost::scoped_ptr<pnl::CInfEngine> onePearlInfEngine(pnl::CPearlInfEngine::Create(mrf2.get()));
		onePearlInfEngine->EnterEvidence(oneEvid.get());

		int query = 0;
		const float eps = 1e-5f;
		for (int i = 0; i < numNodes; ++i)
		{
			query = i;
			oneFGSumMaxInfEngine->MarginalNodes(&query, 1);
			const pnl::CPotential *jpdForFGSumMax = oneFGSumMaxInfEngine->GetQueryJPD();
			oneNaiveInfEngine->MarginalNodes(&query, 1);
			const pnl::CPotential *jpdForNaive = oneNaiveInfEngine->GetQueryJPD();
			onePearlInfEngine->MarginalNodes(&query, 1);
			const pnl::CPotential *jpdForPearl = onePearlInfEngine->GetQueryJPD();

			if (!jpdForPearl->IsFactorsDistribFunEqual(jpdForNaive, eps, 0))
			{
				// TODO [implement] >>
				std::cout << "results of Pearl inference & naive inference are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
			if (!jpdForFGSumMax->IsFactorsDistribFunEqual(jpdForNaive, eps, 0))
			{
				// TODO [implement] >>
				std::cout << "results of sum-max inference & naive inference are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
		}
	}
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/AGaussianMRF2.cpp
pnl::CMRF2 * create_gaussian_mrf2_3()
{
/*
	the model is

	                0
				  /   \
				 /     \
                /       \
			   1         4
			  / \       / \
             /   \     /   \
			2     3   5     6
                           / \
						  /   \
						 7     8

	where
		0, 1, 4, 8 - univariate Gaussian nodes
		2, 5, 6 - bivariate Gaussian nodes
		3, 7 - trivariate Gaussian nodes
*/

	const int numNodes = 9;
	const int numNodeTypes = 3;
	const int numCliques =8;

	pnl::nodeTypeVector nodeTypes(numNodeTypes);
	for (int i = 0; i < numNodeTypes; ++i)
	{
		nodeTypes[i].SetType(false, i + 1);
	}

	pnl::intVector nodeAssociation(numNodes, 0);  // { 0, 0, 1, 2, 0, 1, 1, 2, 0 }
	nodeAssociation[2] = 1;
	nodeAssociation[3] = 2;
	nodeAssociation[5] = 1;
	nodeAssociation[6] = 1;
	nodeAssociation[7] = 2;

	// create graphical model by cliques.
	const std::vector<int> cliqueSizes(numNodes, 2);

	const int clique0[] = { 0, 1 };
	const int clique1[] = { 1, 2 };
	const int clique2[] = { 1, 3 };
	const int clique3[] = { 0, 4 };
	const int clique4[] = { 4, 5 };
	const int clique5[] = { 4, 6 };
	const int clique6[] = { 6, 7 };
	const int clique7[] = { 6, 8 };
	const int *cliques[] = { clique0, clique1, clique2, clique3, clique4, clique5, clique6, clique7 };

	pnl::CMRF2 *mrf2 = pnl::CMRF2::Create(numNodes, numNodeTypes, &nodeTypes.front(), &nodeAssociation.front(), numCliques, &cliqueSizes.front(), cliques);

	// to create factors we need to create their tables
	// create container for factors
	mrf2->AllocFactors();

	// create array of data for every parameter
	const float mean0[] = { 0.6f, 0.4f };  // 2 <- 1 + 1
	const float mean1[] = { 1.5f, 1.5f, 1.7f };  // 3 <- 1 + 2
	const float mean2[] = { 2.1f, 2.9f, 2.3f, 2.7f };  // 4 <- 1 + 3
	const float mean3[] = { 3.1f, 3.9f };  // 2 <- 1 + 1
	const float mean4[] = { 4.2f, 4.8f, 4.3f };  // 3 <- 1 + 2
	const float mean5[] = { 5.4f, 5.6f, 5.6f };  // 3 <- 1 + 2
	const float mean6[] = { 6.8f, 6.2f, 6.9f, 6.1f, 6.7f };  // 5 <- 2 + 3
	const float mean7[] = { 7.1f, 7.4f, 7.3f };  // 3 <- 2 + 1
	const float *means[] = { mean0, mean1, mean2, mean3, mean4, mean5, mean6, mean7 };

	const float cov0[] = { 1.0f, 3.0f, 3.0f, 4.0f };  // 2 x 2
	//const float cov0[] = { 1.0f, 0.0f, 0.0f, 4.0f };  // 2 x 2
	const float cov1[] = { 2.0f, 3.0f, 4.0f, 3.0f, 6.0f, 7.0f, 4.0f, 7.0f, 1.0f };  // 3 x 3
	//const float cov1[] = { 2.0f, 0.0f, 0.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f, 1.0f };  // 3 x 3
	const float cov2[] = { 3.0f, 4.0f, 5.0f, 6.0f, 4.0f, 8.0f, 9.0f, 1.0f, 5.0f, 9.0f, 3.0f, 4.0f, 6.0f, 1.0f, 4.0f, 8.0f };  // 4 x 4
	//const float cov2[] = { 3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 8.0f };  // 4 x 4
	const float cov3[] = { 4.0f, 6.0f, 6.0f, 7.0f };  // 2 x 2
	//const float cov3[] = { 4.0f, 0.0f, 0.0f, 7.0f };  // 2 x 2
	const float cov4[] = { 5.0f, 6.0f, 7.0f, 6.0f, 9.0f, 1.0f, 7.0f, 1.0f, 4.0f };  // 3 x 3
	//const float cov4[] = { 5.0f, 0.0f, 0.0f, 0.0f, 9.0f, 0.0f, 0.0f, 0.0f, 4.0f };  // 3 x 3
	const float cov5[] = { 6.0f, 7.0f, 8.0f, 7.0f, 1.0f, 2.0f, 8.0f, 2.0f, 5.0f };  // 3 x 3
	//const float cov5[] = { 6.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 5.0f };  // 3 x 3
	const float cov6[] = { 7.4f, 7.5f, 7.6f, 7.4f, 7.3f, 7.5f, 7.2f, 7.3f, 7.3f, 7.5f, 7.6f, 7.3f, 7.8f, 7.1f, 7.1f, 7.4f, 7.3f, 7.1f, 7.1f, 7.6f, 7.3f, 7.5f, 7.1f, 7.6f, 7.3f };  // 5 x 5
	//const float cov6[] = { 7.4f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 7.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 7.8f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 7.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 7.3f };  // 5 x 5
	const float cov7[] = { 8.0f, 9.0f, 1.0f, 9.0f, 3.0f, 4.0f, 1.0f, 4.0f, 7.0f };  // 3 x 3
	//const float cov7[] = { 8.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 7.0f };  // 3 x 3
	const float *covs[] = { cov0, cov1, cov2, cov3, cov4, cov5, cov6, cov7 };

	const float coeffs[] = { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

	pnl::CModelDomain *modelDomain = mrf2->GetModelDomain();
	// number of factors is the same as number of cliques - one per clique
	std::vector<pnl::CFactor *> potentials(numCliques, (pnl::CFactor *)NULL);
	for (int i = 0; i < numCliques; ++i)
	{
		potentials[i] = pnl::CGaussianPotential::Create(cliques[i], 2, modelDomain);
		static_cast<pnl::CGaussianPotential *>(potentials[i])->SetCoefficient(coeffs[i], 1);
		potentials[i]->AllocMatrix(means[i], pnl::matH);
		potentials[i]->AllocMatrix(covs[i], pnl::matK);
		mrf2->AttachFactor((pnl::CGaussianPotential *)potentials[i]);
	}

	return mrf2;
}

// [ref] ${PNL_ROOT}/c_pgmtk/tests/src/AGaussianMRF2.cpp
void infer_gaussian_mrf2_using_inference_algorithm_3(const boost::scoped_ptr<pnl::CMRF2> &mrf2)
{
	const int numNodes = mrf2->GetNumberOfNodes();  // TODO [check] >> is it correct?

	{
		const boost::scoped_ptr<pnl::CEvidence> evid(pnl::CEvidence::Create(mrf2.get(), 0, NULL, pnl::valueVector()));

		const boost::scoped_ptr<pnl::CInfEngine> naiveInfEngine(pnl::CNaiveInfEngine::Create(mrf2.get()));
		naiveInfEngine->EnterEvidence(evid.get());

		// belief propagation (Pearl inference)
		const boost::scoped_ptr<pnl::CInfEngine> pearlInfEngine(pnl::CPearlInfEngine::Create(mrf2.get()));
		pearlInfEngine->EnterEvidence(evid.get());

		const boost::scoped_ptr<pnl::CFactorGraph> factorGraph(pnl::CFactorGraph::ConvertFromMNet(mrf2.get()));  // TODO [check] >> is it correct?
		// belief propagation on a factor graph model
		const boost::scoped_ptr<pnl::CInfEngine> fgSumMaxInfEngine(pnl::CFGSumMaxInfEngine::Create(factorGraph.get()));
		fgSumMaxInfEngine->EnterEvidence(evid.get());

		int query = 0;
		const float eps = 1e-3f;
		for (int i = 0; i < numNodes; ++i)
		{
			query = i;
			naiveInfEngine->MarginalNodes(&query, 1);
			const pnl::CPotential *jpdForNaive = naiveInfEngine->GetQueryJPD();
			pearlInfEngine->MarginalNodes(&query, 1);
			const pnl::CPotential *jpdForPearl = pearlInfEngine->GetQueryJPD();
			fgSumMaxInfEngine->MarginalNodes(&query, 1);
			const pnl::CPotential *jpdForFGSumMax = fgSumMaxInfEngine->GetQueryJPD();

			if (!jpdForPearl->IsFactorsDistribFunEqual(jpdForFGSumMax, eps, 0))
			{
				// TODO [implement] >>
				std::cout << "results of Pearl inference & sum-max inference are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
			if (!jpdForNaive->IsFactorsDistribFunEqual(jpdForFGSumMax, eps, 0))
			{
				// TODO [implement] >>
				std::cout << "results of naive inference & sum-max inference are not equal at " << __LINE__ << " in " << __FILE__ << std::endl;
			}
		}
	}
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
			std::cout << "fail to create a probabilistic graphical model at " << __LINE__ << " in " << __FILE__ << std::endl;
			return;
		}

		// belief propagation (Pearl inference) algorithm
		local::infer_simple_mrf2_using_belief_propagation_algorithm(mrf2);  // to be corrected
	}

	// Gaussian pairwise MRF
	std::cout << "\n========== Gaussian pairwise MRF" << std::endl;
	{
		const boost::scoped_ptr<pnl::CMRF2> mrf2_1(local::create_gaussian_mrf2_1());
		const boost::scoped_ptr<pnl::CMRF2> mrf2_2(local::create_gaussian_mrf2_2());
		const boost::scoped_ptr<pnl::CMRF2> mrf2_3(local::create_gaussian_mrf2_3());

		if (!mrf2_1 || !mrf2_2 || !mrf2_3)
		{
			std::cout << "fail to create a probabilistic graphical model at " << __LINE__ << " in " << __FILE__ << std::endl;
			return;
		}

		{
			std::cout << " graph of mixture-of-Gaussians Bayesian network #1";
			mrf2_1->GetGraph()->Dump();
			std::cout << " graph of mixture-of-Gaussians Bayesian network #2";
			mrf2_2->GetGraph()->Dump();
			std::cout << " graph of mixture-of-Gaussians Bayesian network #3";
			mrf2_3->GetGraph()->Dump();
		}

		// belief propagation (Pearl inference) algorithm
		local::infer_gaussian_mrf2_using_inference_algorithm_1(mrf2_1);  // to be corrected
		local::infer_gaussian_mrf2_using_inference_algorithm_2(mrf2_2);  // to be corrected
		local::infer_gaussian_mrf2_using_inference_algorithm_3(mrf2_3);  // to be corrected
	}
}
