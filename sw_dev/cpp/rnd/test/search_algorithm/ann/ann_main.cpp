//----------------------------------------------------------------------
//		File:			ann_sample.cpp
//		Programmer:		Sunil Arya and David Mount
//		Last modified:	03/04/98 (Release 0.1)
//		Description:	Sample program for ANN
//----------------------------------------------------------------------
// Copyright (c) 1997-2005 University of Maryland and Sunil Arya and
// David Mount.  All Rights Reserved.
//
// This software and related documentation is part of the Approximate
// Nearest Neighbor Library (ANN).  This software is provided under
// the provisions of the Lesser GNU Public License (LGPL).  See the
// file ../ReadMe.txt for further information.
//
// The University of Maryland (U.M.) and the authors make no
// representations about the suitability or fitness of this software for
// any purpose.  It is provided "as is" without express or implied
// warranty.
//----------------------------------------------------------------------

//#include "stdafx.h"
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
#include <ANN/ANN.h>					// ANN declarations
#else
#include <ann/ann.h>					// ANN declarations
#endif
#include <cstdlib>						// C standard library
#include <cstdio>						// C I/O (for sscanf)
#include <string>						// string manipulation
#include <fstream>						// file I/O

//----------------------------------------------------------------------
// ann_sample
//
// This is a simple sample program for the ANN library.
// After compiling, it can be run as follows.
//
// ann_sample [-d dim] [-max mpts] [-nn k] [-e eps] [-df data] [-qf query]
//
// where
//		dim				is the dimension of the space (default = 2)
//		mpts			maximum number of data points (default = 1000)
//		k				number of nearest neighbors per query (default 1)
//		eps				is the error bound (default = 0.0)
//		data			file containing data points
//		query			file containing query points
//
// Results are sent to the standard output.
//----------------------------------------------------------------------

namespace {
namespace local {

//----------------------------------------------------------------------
//	Parameters that are set in getArgs()
//----------------------------------------------------------------------
void getArgs(int argc, char *argv[]);			// get command-line arguments

int k = 1;				// number of nearest neighbors
int dim = 2;			// dimension
double eps = 0;			// error bound
int maxPts = 1000;		// maximum number of data points

std::istream *dataIn = NULL;			// input for data points
std::istream *queryIn = NULL;			// input for query points

bool readPt(std::istream &in, ANNpoint p)			// read point (false on EOF)
{
	for (int i = 0; i < dim; ++i)
	{
		if (!(in >> p[i])) return false;
	}
	return true;
}

void printPt(std::ostream &out, ANNpoint p)			// print point
{
	out << "(" << p[0];
	for (int i = 1; i < dim; ++i)
	{
		out << ", " << p[i];
	}
	out << ")" << std::endl;
}

const std::string data1_file = ".\\search_algorithm_data\\ann\\test1-data.pts";
const std::string query1_file = ".\\search_algorithm_data\\ann\\test1-query.pts";
const std::string data2_file = ".\\search_algorithm_data\\ann\\test2-data.pts";
const std::string query2_file = ".\\search_algorithm_data\\ann\\test2-query.pts";

std::ifstream dataStream;				// data file stream
std::ifstream queryStream;				// query file stream

//----------------------------------------------------------------------
//	getArgs - get command line arguments
//----------------------------------------------------------------------

void getArgs(int argc, char *argv[])
{
	if (argc <= 1)								// no arguments
	{
		std::cerr << "Usage:\n\n"
		<< "  ann_sample [-d dim] [-max m] [-nn k] [-e eps] [-df data]"
		   " [-qf query]\n\n"
		<< "  where:\n"
		<< "    dim      dimension of the space (default = 2)\n"
		<< "    m        maximum number of data points (default = 1000)\n"
		<< "    k        number of nearest neighbors per query (default 1)\n"
		<< "    eps      the error bound (default = 0.0)\n"
		<< "    data     name of file containing data points\n"
		<< "    query    name of file containing query points\n\n"
		<< " Results are sent to the standard output.\n"
		<< "\n"
		<< " To run this demo use:\n"
		<< "    ann_sample -df data.pts -qf query.pts\n";
		exit(0);
	}
	int i = 1;
	while (i < argc)							// read arguments
	{
		if (!strcmp(argv[i], "-d"))				// -d option
		{
			local::dim = atoi(argv[++i]);				// get dimension to dump
		}
		else if (!strcmp(argv[i], "-max"))		// -max option
		{
			local::maxPts = atoi(argv[++i]);			// get max number of points
		}
		else if (!strcmp(argv[i], "-nn"))		// -nn option
		{
			local::k = atoi(argv[++i]);				// get number of near neighbors
		}
		else if (!strcmp(argv[i], "-e"))		// -e option
		{
			sscanf(argv[++i], "%lf", &local::eps);		// get error bound
		}
		else if (!strcmp(argv[i], "-df"))		// -df option
		{
			local::dataStream.open(argv[++i], std::ios::in);	// open data file
			if (!local::dataStream)
			{
				std::cerr << "Cannot open data file\n";
				exit(1);
			}
			local::dataIn = &local::dataStream;				// make this the data stream
		}
		else if (!strcmp(argv[i], "-qf"))		// -qf option
		{
			local::queryStream.open(argv[++i], std::ios::in);	// open query file
			if (!local::queryStream)
			{
				std::cerr << "Cannot open query file\n";
				exit(1);
			}
			local::queryIn = &local::queryStream;				// make this query stream
		}
		else									// illegal syntax
		{
			std::cerr << "Unrecognized option.\n";
			exit(1);
		}
		i++;
	}
	if (local::dataIn == NULL || local::queryIn == NULL)
	{
		std::cerr << "-df and -qf options must be specified\n";
		exit(1);
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_ann {

}  // namespace my_ann

int ann_main(int argc, char *argv[])
{
	int					nPts;					// actual number of data points
	ANNpointArray		dataPts;				// data points
	ANNpoint			queryPt;				// query point
	ANNidxArray			nnIdx;					// near neighbor indices
	ANNdistArray		dists;					// near neighbor distances
	ANNkd_tree*			kdTree;					// search structure

	const int mode = 1;
	switch (mode)
	{
	case 0:
		local::getArgs(argc, argv);					// read command-line arguments
		break;
	case 1:
		local::dataStream.open(local::data1_file.c_str(), std::ios::in);
		local::queryStream.open(local::query1_file.c_str(), std::ios::in);
		if (!local::dataStream || !local::queryStream) return 1;
		local::dataIn = &local::dataStream;
		local::queryIn = &local::queryStream;
		local::k = 1;
		local::dim = 2;
		//local::eps = 0.0;
		//local::maxPts = 1000;
		break;
	case 2:
		local::dataStream.open(local::data2_file.c_str(), std::ios::in);
		local::queryStream.open(local::query2_file.c_str(), std::ios::in);
		if (!local::dataStream || !local::queryStream) return 1;
		local::dataIn = &local::dataStream;
		local::queryIn = &local::queryStream;
		local::k = 3;
		local::dim = 8;
		//local::eps = 0.0;
		local::maxPts = 10000;
		break;
	default:
		return 1;
	}

	queryPt = annAllocPt(local::dim);					// allocate query point
	dataPts = annAllocPts(local::maxPts, local::dim);			// allocate data points
	nnIdx = new ANNidx[local::k];						// allocate near neigh indices
	dists = new ANNdist[local::k];						// allocate near neighbor dists

	nPts = 0;									// read data points

	std::cout << "Data Points:\n";
	while (nPts < local::maxPts && local::readPt(*local::dataIn, dataPts[nPts]))
	{
		local::printPt(std::cout, dataPts[nPts]);
		nPts++;
	}

	kdTree = new ANNkd_tree(					// build search structure
					dataPts,					// the data points
					nPts,						// number of points
					local::dim);						// dimension of space

	while (local::readPt(*local::queryIn, queryPt))			// read query points
	{
		std::cout << "Query point: ";				// echo query point
		local::printPt(std::cout, queryPt);

		kdTree->annkSearch(						// search
				queryPt,						// query point
				local::k,								// number of near neighbors
				nnIdx,							// nearest neighbors (returned)
				dists,							// distance (returned)
				local::eps);							// error bound

		std::cout << "\tNN:\tIndex\tDistance\n";
		for (int i = 0; i < local::k; ++i)				// print summary
		{
			dists[i] = sqrt(dists[i]);			// unsquare distance
			std::cout << "\t" << i << "\t" << nnIdx[i] << "\t" << dists[i] << std::endl;
		}
	}
    delete [] nnIdx;							// clean things up
    delete [] dists;
    delete kdTree;
	annClose();									// done with ANN

	return EXIT_SUCCESS;
}
