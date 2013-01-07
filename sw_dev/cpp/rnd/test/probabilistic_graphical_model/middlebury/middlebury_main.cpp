//#define COUNT_TRUNCATIONS 1
#include "../middlebury_lib/mrf.h"
#include "../middlebury_lib/ICM.h"
#include "../middlebury_lib/GCoptimization.h"
#include "../middlebury_lib/MaxProdBP.h"
#include "../middlebury_lib/TRW-S.h"
#include "../middlebury_lib/BP-S.h"
#include <boost/smart_ptr.hpp>
#include <vector>
#include <iostream>


namespace {
namespace local {

// using fixed (array) smoothness cost (for 4-connected grid)
void generateEnergyFunction_DataCostArray_FixedSmoothnessCostArray(const int sizeX, const int sizeY, const int numLabels, std::vector<MRF::CostVal> &D, std::vector<MRF::CostVal> &V, std::vector<MRF::CostVal> &horzWeight, std::vector<MRF::CostVal> &vertWeight, boost::shared_ptr<DataCost> &dataCost, boost::shared_ptr<SmoothnessCost> &smoothnessCost)
{
	// generate function
	int i, j;
	for (i = 0; i < numLabels; ++i)
		for (j = i; j < numLabels; ++j)
			V[i*numLabels+j] = V[j*numLabels+i] = (MRF::CostVal)((i == j) ? 0 : 2.3);

	for (std::vector<MRF::CostVal>::iterator it = D.begin(); it != D.end(); ++it) *it = ((MRF::CostVal)(std::rand() % 100)) / 10 + 1;
	for (std::vector<MRF::CostVal>::iterator it = horzWeight.begin(); it != horzWeight.end(); ++it) *it = std::rand() % 3 + 1;
	for (std::vector<MRF::CostVal>::iterator it = vertWeight.begin(); it != vertWeight.end(); ++it) *it = std::rand() % 3 + 1;

	// allocate energy
	dataCost.reset(new DataCost(&D[0]));
	smoothnessCost.reset(new SmoothnessCost(&V[0], &horzWeight[0], &vertWeight[0]));
}

// using truncated linear smoothness cost (for 4-connected grid)
void generateEnergyFunction_DataCostArray_LinearTruncatedSmoothnessCostArray(const int sizeX, const int sizeY, const int numLabels, std::vector<MRF::CostVal> &D, std::vector<MRF::CostVal> &horzWeight, std::vector<MRF::CostVal> &vertWeight, boost::shared_ptr<DataCost> &dataCost, boost::shared_ptr<SmoothnessCost> &smoothnessCost)
{
	// generate function
	for (std::vector<MRF::CostVal>::iterator it = D.begin(); it != D.end(); ++it) *it = ((MRF::CostVal)(std::rand() % 100)) / 10 + 1;
	for (std::vector<MRF::CostVal>::iterator it = horzWeight.begin(); it != horzWeight.end(); ++it) *it = std::rand() % 3;
	for (std::vector<MRF::CostVal>::iterator it = vertWeight.begin(); it != vertWeight.end(); ++it) *it = std::rand() % 3;
	MRF::CostVal smoothMax = (MRF::CostVal)25.5, lambda = (MRF::CostVal)2.7;

	// allocate energy
	dataCost.reset(new DataCost(&D[0]));
	smoothnessCost.reset(new SmoothnessCost(1, smoothMax, lambda, &horzWeight[0], &vertWeight[0]));
}

// using truncated quadratic smoothness cost (for 4-connected grid)
void generateEnergyFunction_DataCostArray_QuadraticTruncatedSmoothnessCostArray(const int sizeX, const int sizeY, const int numLabels, std::vector<MRF::CostVal> &D, std::vector<MRF::CostVal> &horzWeight, std::vector<MRF::CostVal> &vertWeight, boost::shared_ptr<DataCost> &dataCost, boost::shared_ptr<SmoothnessCost> &smoothnessCost)
{
	// generate function
	for (std::vector<MRF::CostVal>::iterator it = D.begin(); it != D.end(); ++it) *it = ((MRF::CostVal)(std::rand() % 100)) / 10 + 1;
	for (std::vector<MRF::CostVal>::iterator it = horzWeight.begin(); it != horzWeight.end(); ++it) *it = std::rand() % 3;
	for (std::vector<MRF::CostVal>::iterator it = vertWeight.begin(); it != vertWeight.end(); ++it) *it = std::rand() % 3;
	MRF::CostVal smoothMax = (MRF::CostVal)5.5, lambda = (MRF::CostVal)2.7;

	// allocate energy
	dataCost.reset(new DataCost(&D[0]));
	smoothnessCost.reset(new SmoothnessCost(2, smoothMax, lambda, &horzWeight[0], &vertWeight[0]));
}

MRF::CostVal MyDataCostFunction(int pix, int i)
{
	return ((pix*i + i + pix) % 30) / ((MRF::CostVal)3);
}

MRF::CostVal MySmoothnessCostFunction(int pix1, int pix2, int i, int j)
{
	if (pix2 < pix1)  // ensure that fnCost(pix1, pix2, i, j) == fnCost(pix2, pix1, j, i)
	{
		int tmp;
		tmp = pix1; pix1 = pix2; pix2 = tmp; 
		tmp = i; i = j; j = tmp;
	}
	MRF::CostVal answer = (pix1*(i+1)*(j+2) + pix2*i*j*pix1 - 2*i*j*pix1) % 100;
	return answer / 10;
}

// using general smoothness functions (for 4-connected grid)
void generateEnergyFunction_DataCostFunction_GeneralSmoothnessCostFunction(boost::shared_ptr<DataCost> &dataCost, boost::shared_ptr<SmoothnessCost> &smoothnessCost)
{
	dataCost.reset(new DataCost(MyDataCostFunction));
	smoothnessCost.reset(new SmoothnessCost(MySmoothnessCostFunction));
}

// iterated conditional modes (ICM) algorithm
void ICM_algorithm(const int sizeX, const int sizeY, const int numLabels, EnergyFunction *energyFunc)
{
	boost::scoped_ptr<MRF> mrf(new ICM(sizeX, sizeY, numLabels, energyFunc));
	
	mrf->initialize();
	mrf->clearAnswer();

	MRF::EnergyVal E = mrf->totalEnergy();
	std::cout << "energy at the start = " << (float)E << " (" << (float)mrf->smoothnessEnergy() << ',' << (float)mrf->dataEnergy() << ')' << std::endl;

	float time, totalElapsedTime = 0.0;
	for (int iter = 0; iter < 6; ++iter)
	{
		mrf->optimize(10, time);

		E = mrf->totalEnergy();
		totalElapsedTime = totalElapsedTime + time;
		std::cout << "energy = " << (float)E << " (" << totalElapsedTime << " secs)" << std::endl;
	}
}

// graph cuts with expansion algorithm
void graph_cuts_with_expansion_algorithm(const int sizeX, const int sizeY, const int numLabels, EnergyFunction *energyFunc)
{
	boost::scoped_ptr<MRF> mrf(new Expansion(sizeX, sizeY, numLabels, energyFunc));

/*
	// for general neighborhood systems
    for (int i = 0; i < sizeX; ++i)
        for (int j = 0; j < sizeY; ++j)
			// sets up a full neighborhood system with weights w_pq= (p-q)^2
            mrf->setNeighbors(i, j, (i-j) * (i-j));
*/

	mrf->initialize();
	mrf->clearAnswer();

	MRF::EnergyVal E = mrf->totalEnergy();
	std::cout << "energy at the start = " << (float)E << " (" << (float)mrf->smoothnessEnergy() << ',' << (float)mrf->dataEnergy() << ')' << std::endl;

#ifdef COUNT_TRUNCATIONS
	int truncCnt = 0, totalCnt = 0;
#endif

	float time, totalElapsedTime = 0.0;
	for (int iter = 0; iter < 6; ++iter)
	{
		mrf->optimize(1, time);

		E = mrf->totalEnergy();
		totalElapsedTime = totalElapsedTime + time;
		std::cout << "energy = " << (float)E << " (" << totalElapsedTime << " secs)" << std::endl;
	}

#ifdef COUNT_TRUNCATIONS
	if (truncCnt > 0)
		std::cout << "***WARNING: " << truncCnt << " terms (" << float(100.0 * truncCnt / totalCnt) << "%%) were truncated to ensure regularity" << std::endl;
#endif
}

// graph cuts with swap algorithm
void graph_cuts_with_swap_algorithm(const int sizeX, const int sizeY, const int numLabels, EnergyFunction *energyFunc)
{
	boost::scoped_ptr<MRF> mrf(new Swap(sizeX, sizeY, numLabels, energyFunc));

/*
	// for general neighborhood systems
    for (int i = 0; i < sizeX; ++i)
        for (int j = 0; j < sizeY; ++j)
			// sets up a full neighborhood system with weights w_pq= (p-q)^2
            mrf->setNeighbors(i, j, (i-j) * (i-j));
*/

	mrf->initialize();
	mrf->clearAnswer();

	MRF::EnergyVal E = mrf->totalEnergy();
	std::cout << "energy at the start = " << (float)E << " (" << (float)mrf->smoothnessEnergy() << ',' << (float)mrf->dataEnergy() << ')' << std::endl;

#ifdef COUNT_TRUNCATIONS
	int truncCnt = 0, totalCnt = 0;
#endif

	float time, totalElapsedTime = 0.0;
	for (int iter = 0; iter < 8; ++iter)
	{
		mrf->optimize(1, time);

		E = mrf->totalEnergy();
		totalElapsedTime = totalElapsedTime + time;
		std::cout << "energy = " << (float)E << '(' << totalElapsedTime << " secs)" << std::endl;
	}

#ifdef COUNT_TRUNCATIONS
	if (truncCnt > 0)
		std::cout << "***WARNING: " << truncCnt << " terms (" << float(100.0 * truncCnt / totalCnt) << "%%) were truncated to ensure regularity" << std::endl;
#endif
}

// max-product (loopy) belief propagation (BP) algorithm: Pearl algorithm
void max_product_BP_algorithm(const int sizeX, const int sizeY, const int numLabels, EnergyFunction *energyFunc)
{
	boost::scoped_ptr<MRF> mrf(new MaxProdBP(sizeX, sizeY, numLabels, energyFunc));

	mrf->initialize();
	mrf->clearAnswer();

	MRF::EnergyVal E = mrf->totalEnergy();
	std::cout << "energy at the start = " << (float)E << '(' << (float)mrf->smoothnessEnergy() << ',' << (float)mrf->dataEnergy() << ')' << std::endl;

	float time, totalElapsedTime = 0.0;
	for (int iter = 0; iter < 10; ++iter)
	{
		mrf->optimize(1, time);

		E = mrf->totalEnergy();
		totalElapsedTime = totalElapsedTime + time;
		std::cout << "energy = " << (float)E << " (" << totalElapsedTime << " secs)" << std::endl;
	}
}

// sequential tree reweighted (TRW) message passing algorithm
void TRW_S_algorithm(const int sizeX, const int sizeY, const int numLabels, EnergyFunction *energyFunc)
{
	boost::scoped_ptr<MRF> mrf(new TRWS(sizeX, sizeY, numLabels, energyFunc));

	// can disable caching of values of general smoothness function
	//mrf->dontCacheSmoothnessCosts();

	mrf->initialize();
	mrf->clearAnswer();

	MRF::EnergyVal E = mrf->totalEnergy();
	std::cout << "energy at the start = " << (float)E << " (" << (float)mrf->smoothnessEnergy() << ',' << (float)mrf->dataEnergy() << ')' << std::endl;

	float time, totalElapsedTime = 0.0;
	for (int iter = 0; iter < 10; ++iter)
	{
		mrf->optimize(10, time);

		E = mrf->totalEnergy();
		const double lowerBound = mrf->lowerBound();
		totalElapsedTime = totalElapsedTime + time;
		std::cout << "energy = " << (float)E << ", lower bound = " << lowerBound << " (" << totalElapsedTime << " secs)" << std::endl;
	}
}

// BP-S
void BP_S_algorithm(const int sizeX, const int sizeY, const int numLabels, EnergyFunction *energyFunc)
{
	boost::scoped_ptr<MRF> mrf(new BPS(sizeX, sizeY, numLabels, energyFunc));

	// can disable caching of values of general smoothness function:
	//mrf->dontCacheSmoothnessCosts();

	mrf->initialize();
	mrf->clearAnswer();

	MRF::EnergyVal E = mrf->totalEnergy();
	std::cout << "energy at the start = " << (float)E << " (" << (float)mrf->smoothnessEnergy() << ',' << (float)mrf->dataEnergy() << ')' << std::endl;

	float time, totalElapsedTime = 0.0;
	for (int iter = 0; iter < 10; ++iter)
	{
		mrf->optimize(10, time);

		E = mrf->totalEnergy();
		totalElapsedTime = totalElapsedTime + time;
		std::cout << "energy = " << (float)E << " (" << totalElapsedTime << " secs)" << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace middlebury {

}  // namespace middlebury

int middlebury_main(int argc, char *argv[])
{
	const int sizeX = 50;
	const int sizeY = 50;
	const int numLabels = 20;

	// 1. set up an energy function
	std::vector<MRF::CostVal> D(sizeX * sizeY * numLabels, (MRF::CostVal)0);
	std::vector<MRF::CostVal> V(numLabels * numLabels, (MRF::CostVal)0);
	std::vector<MRF::CostVal> horzWeight(sizeX * sizeY, (MRF::CostVal)0);
	std::vector<MRF::CostVal> vertWeight(sizeX * sizeY, (MRF::CostVal)0);

	// set up data cost term
	boost::shared_ptr<DataCost> dataCost;
	// set up smoothness cost term
	boost::shared_ptr<SmoothnessCost> smoothnessCost;

	local::generateEnergyFunction_DataCostArray_FixedSmoothnessCostArray(sizeX, sizeY, numLabels, D, V, horzWeight, vertWeight, dataCost, smoothnessCost);
	//local::generateEnergyFunction_DataCostArray_LinearTruncatedSmoothnessCostArray(sizeX, sizeY, numLabels, D, horzWeight, vertWeight, dataCost, smoothnessCost);
	//local::generateEnergyFunction_DataCostArray_QuadraticTruncatedSmoothnessCostArray(sizeX, sizeY, numLabels, D, horzWeight, vertWeight, dataCost, smoothnessCost);
	//local::generateEnergyFunction_DataCostFunction_GeneralSmoothnessCostFunction(dataCost, smoothnessCost);

	// allocate energy
	boost::scoped_ptr<EnergyFunction> energyFunc(new EnergyFunction(dataCost.get(), smoothnessCost.get()));

	// 2. invoke an optimization algorithm
	std::cout << "****** iterated conditional modes (ICM) algorithm" << std::endl;
	local::ICM_algorithm(sizeX, sizeY, numLabels, energyFunc.get());

	std::cout << "****** graph cuts with expansion algorithm" << std::endl;
	local::graph_cuts_with_expansion_algorithm(sizeX, sizeY, numLabels, energyFunc.get());

	std::cout << "****** graph cuts with swap algorithm" << std::endl;
	local::graph_cuts_with_swap_algorithm(sizeX, sizeY, numLabels, energyFunc.get());

	std::cout << "****** max-product (loopy) belief propagation (BP) algorithm" << std::endl;
	local::max_product_BP_algorithm(sizeX, sizeY, numLabels, energyFunc.get());

	std::cout << "****** sequential tree reweighted (TRW) message passing algorithm" << std::endl;
	local::TRW_S_algorithm(sizeX, sizeY, numLabels, energyFunc.get());

	std::cout << "****** BP-S" << std::endl;
	local::BP_S_algorithm(sizeX, sizeY, numLabels, energyFunc.get());

	return 0;
}
