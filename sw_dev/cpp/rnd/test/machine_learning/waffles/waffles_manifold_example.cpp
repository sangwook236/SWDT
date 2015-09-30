//#include "stdafx.h"
#include <GClasses/GManifold.h>
#include <GClasses/GNeighborFinder.h>
#include <GClasses/GMatrix.h>
#include <GClasses/GBits.h>
#include <GClasses/GFile.h>
#include <GClasses/GTime.h>
#include <GClasses/GHolders.h>
#include <GClasses/GThread.h>
#include <GClasses/GVec.h>
#include <GClasses/GMath.h>
#include <GClasses/GHillClimber.h>
#include <GClasses/GHeap.h>
#include <GClasses/GApp.h>
#include <iostream>
#include <cmath>


namespace {
namespace local {

double LengthOfSineFunc(void *pThis, double x)
{
	double d = std::cos(x);
	return std::sqrt(d * d + 1.0);
}

double LengthOfSwissRoll(double x)
{
#ifdef WINDOWS
	throw Ex("not implemented yet for Windows");
	return 0;
#else
	return (x * std::sqrt(x * x + 1) + std::asinh(x)) / 2;
#endif
}

void generate_swiss_roll_data(const std::size_t nPoints, const bool bComputeIdeal, GClasses::GMatrix *&pData, GClasses::GRelation *&pRelation, double *&pIdealResults, GClasses::GRand *prng)
{
    // Make the relation.
    pRelation = new GClasses::GUniformRelation(3, 0);

    // Make the ARFF data.
    if (bComputeIdeal)
        pIdealResults = new double [nPoints * 2];
    pData = new GClasses::GMatrix(pRelation->clone());
    pData->reserve(nPoints);

    //
    for (std::size_t n = 0; n < nPoints; ++n)
    {
        const double t = ((double)n * 8) / nPoints;
        double *pVector = pData->newRow();
        pVector[0] = (t + 2) * std::sin(t) + 14;
        pVector[1] = prng->uniform() * 12 - 6;
        pVector[2] = (t + 2) * std::cos(t);
        if (bComputeIdeal)
        {
            pIdealResults[2 * n] = pVector[1];
            pIdealResults[2 * n + 1] = LengthOfSwissRoll(t + 2);/* - LengthOfSwissRoll(2);*/
        }
    }
}

void generate_s_curve_data(const std::size_t nPoints, const bool bComputeIdeal, GClasses::GMatrix *&pData, GClasses::GRelation *&pRelation, double *&pIdealResults, GClasses::GRand *prng)
{
    // Make the relation.
    pRelation = new GClasses::GUniformRelation(3, 0);

    // Make the ARFF data.
    if (bComputeIdeal)
        pIdealResults = new double [nPoints * 2];
    pData = new GClasses::GMatrix(pRelation->clone());
    pData->reserve(nPoints);

    //
    for (std::size_t n = 0; n < nPoints; ++n)
    {
        const double t = ((double)n * 2.2 * M_PI - .1 * M_PI) / nPoints;
        double *pVector = pData->newRow();
        pVector[0] = 1.0 - std::sin(t);
        pVector[1] = t;
        pVector[2] = prng->uniform() * 2;
        if (bComputeIdeal)
        {
            pIdealResults[2 * n] = pVector[2];
            pIdealResults[2 * n + 1] = (n > 0 ? GClasses::GMath::integrate(LengthOfSineFunc, 0, t, n + 30, NULL) : 0);
        }
    }
}

void generate_spirals_data(const std::size_t nPoints, const bool bComputeIdeal, GClasses::GMatrix *&pData, GClasses::GRelation *&pRelation, double *&pIdealResults, GClasses::GRand *prng)
{
    // Make the relation.
    pRelation = new GClasses::GUniformRelation(3, 0);

    // Make the ARFF data.
    if (bComputeIdeal)
        pIdealResults = new double [nPoints * 1];
    pData = new GClasses::GMatrix(pRelation->clone());
    pData->reserve(nPoints);

    //
    const double dHeight = 3;
    const double dWraps = 1.5;
    const double dSpiralLength = std::sqrt((dWraps * 2.0 * M_PI) * (dWraps * 2.0 * M_PI) + dHeight * dHeight);
    const double dTotalLength = 2.0 * (dSpiralLength + 1); // radius = 1
    for (std::size_t n = 0; n < nPoints; ++n)
    {
        const double t = ((double)n * dTotalLength) / nPoints;
        double *pVector = pData->newRow();
        if (t < dSpiralLength)
        {
            const double d = (dSpiralLength - t) * dWraps * 2 * M_PI / dSpiralLength; // d = radians
            pVector[0] = -std::cos(d);
            pVector[1] = dHeight * t / dSpiralLength;
            pVector[2] = -std::sin(d);
        }
        else if (t - 2.0 - dSpiralLength >= 0)
        {
            const double d = (t - 2.0 - dSpiralLength) * dWraps * 2 * M_PI / dSpiralLength; // d = radians
            pVector[0] = std::cos(d);
            pVector[1] = dHeight * (dSpiralLength - (t - 2.0 - dSpiralLength)) / dSpiralLength;
            pVector[2] = std::sin(d);
        }
        else
        {
            const double d = (t - dSpiralLength) / 2.0; // 2 = diameter
            pVector[0] = 2.0 * d - 1.0;
            pVector[1] = dHeight;
            pVector[2] = 0;
        }
        if (bComputeIdeal)
            pIdealResults[n] = dTotalLength * n / nPoints;
    }
}

void manifold_sculpting_for_swiss_roll()
{
    GClasses::GRand rng(0);

    const std::size_t nPoints = 2000;
    const int nNeighbors = 20;
    const double dSquishingRate = 0.98;
    const bool bComputeIdeal = false;
    const std::size_t nTargetDims = 2;

    // Generate data.
    GClasses::GMatrix *pData = NULL;
    GClasses::GRelation *pRelation = NULL;
    double *pIdealResults = NULL;
    generate_swiss_roll_data(nPoints, bComputeIdeal, pData, pRelation, pIdealResults, &rng);

    // Create model.
    GClasses::GManifoldSculpting *pSculpter = new GClasses::GManifoldSculpting(nNeighbors, nTargetDims, &rng);
    pSculpter->beginTransform(pData);
    pSculpter->setSquishingRate(dSquishingRate);

    // Learn model.
    std::cout << "start manifold sculpting for swiss roll ..." << std::endl;
    {
        const double timeStart = GClasses::GTime::seconds();

        const int nDataPoints = pSculpter->data().rows();
        const std::size_t MAX_ITERATIONS = 2000;
        std::size_t iter = 0;
        bool bConverged = false;
        while (++iter < MAX_ITERATIONS)
        {
            pSculpter->squishPass((std::size_t)rng.next(nDataPoints));

            if (pSculpter->learningRate() / pSculpter->aveNeighborDist() < .001)
            {
                bConverged = true;
                break;
            }
        }

        const double timeEnd = GClasses::GTime::seconds();
        std::cout << "\tconverged = " << (bConverged ? "true" : "false") << ", elapsed time = " << (timeEnd - timeStart) << std::endl;
    }
    std::cout << "end manifold sculpting for swiss roll ..." << std::endl;

    // Clean up.
    delete pSculpter;
    delete pData;
    delete pRelation;
    delete [] pIdealResults;
}

void manifold_sculpting_for_s_curve()
{
    GClasses::GRand rng(0);

    const std::size_t nPoints = 2000;
    const int nNeighbors = 20;
    const double dSquishingRate = 0.98;
    const bool bComputeIdeal = false;
    const std::size_t nTargetDims = 2;

    // Generate data.
    GClasses::GMatrix *pData = NULL;
    GClasses::GRelation *pRelation = NULL;
    double *pIdealResults = NULL;
    generate_s_curve_data(nPoints, bComputeIdeal, pData, pRelation, pIdealResults, &rng);

    // Generate data.
    GClasses::GManifoldSculpting *pSculpter = new GClasses::GManifoldSculpting(nNeighbors, nTargetDims, &rng);
    pSculpter->beginTransform(pData);
    pSculpter->setSquishingRate(dSquishingRate);

    // Learn model.
    std::cout << "start manifold sculpting for S-curve ..." << std::endl;
    {
        const double timeStart = GClasses::GTime::seconds();

        const int nDataPoints = pSculpter->data().rows();
        const std::size_t MAX_ITERATIONS = 2000;
        std::size_t iter = 0;
        bool bConverged = false;
        while (++iter < MAX_ITERATIONS)
        {
            pSculpter->squishPass((std::size_t)rng.next(nDataPoints));

            if (pSculpter->learningRate() / pSculpter->aveNeighborDist() < .001)
            {
                bConverged = true;
                break;
            }
        }

        const double timeEnd = GClasses::GTime::seconds();
        std::cout << "\tconverged = " << (bConverged ? "true" : "false") << ", elapsed time = " << (timeEnd - timeStart) << std::endl;
    }
    std::cout << "end manifold sculpting for S-curve ..." << std::endl;

    // Clean up.
    delete pSculpter;
    delete pData;
    delete pRelation;
    delete [] pIdealResults;
}

void manifold_sculpting_for_spirals()
{
    GClasses::GRand rng(0);

    const std::size_t nPoints = 2000;
    const int nNeighbors = 20;
    const double dSquishingRate = 0.98;
    const bool bComputeIdeal = false;
    const std::size_t nTargetDims = 2;

    // Generate data.
    GClasses::GMatrix *pData = NULL;
    GClasses::GRelation *pRelation = NULL;
    double *pIdealResults = NULL;
    generate_spirals_data(nPoints, bComputeIdeal, pData, pRelation, pIdealResults, &rng);

    // Create model.
    GClasses::GManifoldSculpting *pSculpter = new GClasses::GManifoldSculpting(nNeighbors, nTargetDims, &rng);
    pSculpter->beginTransform(pData);
    pSculpter->setSquishingRate(dSquishingRate);

    // Leanr model.
    std::cout << "start manifold sculpting for spirals ..." << std::endl;
    {
        const double timeStart = GClasses::GTime::seconds();

        const int nDataPoints = pSculpter->data().rows();
        const std::size_t MAX_ITERATIONS = 2000;
        std::size_t iter = 0;
        bool bConverged = false;
        while (++iter < MAX_ITERATIONS)
        {
            pSculpter->squishPass((std::size_t)rng.next(nDataPoints));

            if (pSculpter->learningRate() / pSculpter->aveNeighborDist() < .001)
            {
                bConverged = true;
                break;
            }
        }

        const double timeEnd = GClasses::GTime::seconds();
        std::cout << "\tconverged = " << (bConverged ? "true" : "false") << ", elapsed time = " << (timeEnd - timeStart) << std::endl;
    }
    std::cout << "end manifold sculpting for spirals ..." << std::endl;

    // Clean up.
    delete pSculpter;
    delete pData;
    delete pRelation;
    delete [] pIdealResults;
}

void semi_supervised_manifold_sculpting_for_swiss_roll()
{
    GClasses::GRand rng(0);

    const std::size_t nPoints = 2000;
    const int nNeighbors = 40;
    const double dSquishingRate = 0.98;
    const bool bComputeIdeal = false;
    const int nSupervisedPoints = 100;
    const std::size_t nTargetDims = 2;

    // Generate data.
    GClasses::GMatrix *pData = NULL;
    GClasses::GRelation *pRelation = NULL;
    double *pIdealResults = NULL;
    generate_swiss_roll_data(nPoints, bComputeIdeal, pData, pRelation, pIdealResults, &rng);

    // Create previous model.
    GClasses::GManifoldSculpting *pPrevSculpter = new GClasses::GManifoldSculpting(nNeighbors, nTargetDims, &rng);
    pPrevSculpter->beginTransform(pData);
    pPrevSculpter->setSquishingRate(dSquishingRate);

    // Learn previous model.
    std::cout << "start manifold sculpting for swiss roll ..." << std::endl;
    {
        const double timeStart = GClasses::GTime::seconds();

        const int nDataPoints = pPrevSculpter->data().rows();
        const std::size_t MAX_ITERATIONS = 2000;
        std::size_t iter = 0;
        bool bConverged = false;
        while (++iter < MAX_ITERATIONS)
        {
            pPrevSculpter->squishPass((std::size_t)rng.next(nDataPoints));

            if (pPrevSculpter->learningRate() / pPrevSculpter->aveNeighborDist() < .001)
            {
                bConverged = true;
                break;
            }
        }

        const double timeEnd = GClasses::GTime::seconds();
        std::cout << "\tconverged = " << (bConverged ? "true" : "false") << ", elapsed time = " << (timeEnd - timeStart) << std::endl;
    }
    std::cout << "end manifold sculpting for swiss roll ..." << std::endl;

    // Generate data.
    rng.setSeed(0);
    generate_swiss_roll_data(nPoints, bComputeIdeal, pData, pRelation, pIdealResults, &rng);

    // Create model.
    GClasses::GManifoldSculpting *pSculpter = new GClasses::GManifoldSculpting(nNeighbors, nTargetDims, &rng);
    pSculpter->beginTransform(pData);
    pSculpter->setSquishingRate(dSquishingRate);

    // Set the supervised points.
    {
        for (int i = 0; i < nSupervisedPoints; ++i)
        {
            const std::size_t nPoint = (std::size_t)rng.next(nPoints);
            GClasses::GVec::copy(pSculpter->data().row(nPoint), pPrevSculpter->data().row(nPoints), pSculpter->data().relation().size());
            pSculpter->clampPoint(nPoint);
        }
    }

    delete pPrevSculpter;
    pPrevSculpter = NULL;

    // Learn model.
    std::cout << "start semi-supervised manifold sculpting for swiss roll ..." << std::endl;
    {
        const double timeStart = GClasses::GTime::seconds();

        const int nDataPoints = pSculpter->data().rows();
        const std::size_t MAX_ITERATIONS = 2000;
        std::size_t iter = 0;
        bool bConverged = false;
        while (++iter < MAX_ITERATIONS)
        {
            pSculpter->squishPass((std::size_t)rng.next(nDataPoints));

            if (pSculpter->learningRate() / pSculpter->aveNeighborDist() < .001)
            {
                bConverged = true;
                break;
            }
        }

        const double timeEnd = GClasses::GTime::seconds();
        std::cout << "\tconverged = " << (bConverged ? "true" : "false") << ", elapsed time = " << (timeEnd - timeStart) << std::endl;
    }
    std::cout << "end semi-supervised manifold sculpting for swiss roll ..." << std::endl;

    // Clean up.
    delete pSculpter;
    delete pData;
    delete pRelation;
    delete [] pIdealResults;
}

// REF [function] >> DoFaceDemo() in ${WAFFLES_HOME}/demos/manifold/src/main.cpp.
void manifold_sculpting_for_face()
{
    throw std::runtime_error("not yet implemented");
}

void dimensionality_reduction(const int idAlgorithm, const int idData)
{
    GClasses::GRand rng(0);

    const std::size_t nPoints = 2000;
    const int nNeighbors = 14;
    const bool bComputeIdeal = false;
    const std::size_t nTargetDims = 2;

    // Generate data.
    GClasses::GMatrix *pData = NULL;
    GClasses::GRelation *pRelation = NULL;
    double *pIdealResults = NULL;

    switch (idData)
    {
    case 1:
        generate_swiss_roll_data(nPoints, bComputeIdeal, pData, pRelation, pIdealResults, &rng);
        break;
    case 2:
        generate_s_curve_data(nPoints, bComputeIdeal, pData, pRelation, pIdealResults, &rng);
        break;
    case 3:
        generate_spirals_data(nPoints, bComputeIdeal, pData, pRelation, pIdealResults, &rng);
        break;
    default:
        throw std::runtime_error("invalid data ID");
        break;
    }

    // Create model.
    GClasses::GTransform *pDimensionalityReducer = NULL;
    GClasses::GNeighborFinder *pNeighberFinder = new GClasses::GBallTree(pData, nNeighbors);
    switch (idAlgorithm)
    {
    case 1:
        pDimensionalityReducer = new GClasses::GLLE(nNeighbors, nTargetDims, &rng);
        if (pNeighberFinder) ((GClasses::GLLE *)pDimensionalityReducer)->setNeighborFinder(pNeighberFinder);
        break;
    case 2:
        pDimensionalityReducer = new GClasses::GIsomap(nNeighbors, nTargetDims, &rng);
        if (pNeighberFinder) ((GClasses::GIsomap *)pDimensionalityReducer)->setNeighborFinder(pNeighberFinder);
        break;
    case 3:
        {
            const std::size_t reps = 1;  // the number of times to compute the embedding. If you just want fast results, use reps = 1.
            pDimensionalityReducer = new GClasses::GBreadthFirstUnfolding(reps, nNeighbors, nTargetDims);
            if (pNeighberFinder) ((GClasses::GBreadthFirstUnfolding *)pDimensionalityReducer)->setNeighborFinder(pNeighberFinder);
            //((GClasses::GBreadthFirstUnfolding *)pDimensionalityReducer)->useMds(true);
        }
        break;
    case 4:
        pDimensionalityReducer = new GClasses::GNeuroPCA(nTargetDims, &rng);
        break;
    case 5:
        pDimensionalityReducer = new GClasses::GScalingUnfolder();
        ((GClasses::GScalingUnfolder *)pDimensionalityReducer)->setNeighborCount(nNeighbors);
        ((GClasses::GScalingUnfolder *)pDimensionalityReducer)->setTargetDims(nTargetDims);
        //((GClasses::GScalingUnfolder *)pDimensionalityReducer)->setPasses();  // the number of times to 'scale the data then recover local relationships.
        //((GClasses::GScalingUnfolder *)pDimensionalityReducer)->setRefinesPerScale();  // the number of times to refine the points after each scaling.
        //((GClasses::GScalingUnfolder *)pDimensionalityReducer)->setScaleRate();  // the scaling rate. The default is 0.9.
        //((GClasses::GScalingUnfolder *)pDimensionalityReducer)->unfold();
        break;
    default:
        throw std::runtime_error("invalid dimensionality reduction algorithm ID");
        break;
    }

    // Learn model.
    std::cout << "start dimensionality reduction ..." << std::endl;
    {
        const double timeStart = GClasses::GTime::seconds();

        GClasses::GMatrix *pReducedData = pDimensionalityReducer->reduce(*pData);

        const double timeEnd = GClasses::GTime::seconds();
        std::cout << "\telapsed time = " << (timeEnd - timeStart) << std::endl;

        //
        delete pReducedData;
    }
    std::cout << "end dimensionality reduction ..." << std::endl;

    // Clean up.
    delete pDimensionalityReducer;
    delete pData;
    delete pRelation;
    delete [] pIdealResults;
}

}  // namespace local
}  // unnamed namespace

namespace my_waffles {

void dimensionality_reduction()
{
    const int idAlgorithm = 3;
    const int idData = 1;
    local::dimensionality_reduction(idAlgorithm, idData);
}

// REF [file] >> ${WAFFLES_HOME}/demos/manifold/src/main.cpp.
void manifold_sculpting_example()
{
    local::manifold_sculpting_for_swiss_roll();
    local::manifold_sculpting_for_s_curve();
    local::manifold_sculpting_for_spirals();

    local::semi_supervised_manifold_sculpting_for_swiss_roll();

    //local::manifold_sculpting_for_face();  // not yet implemented.
}

}  // namespace my_waffles
