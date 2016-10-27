//#include "stdafx.h"
#include "../slash_lib/lsh.h"
#include "../slash_lib/slsh.h"
#include "../slash_lib/bitvector64.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>


namespace {
namespace local {

#define BAD_LINEAR_SEARCH_FRACTION 1e-2

#define NPOINTS 1e5
#define NQUERIES 1e5

int d = 64, k = 6, L = 2, limit = 10;
slash::SLSH<BitVector64> *slsh;
slash::LSH<BitVector64, slash::SLSH<BitVector64> > *lsh;
char buf[256];
std::vector<BitVector64> points;

void init()
{
	srandom((unsigned int)time(NULL));

	for (size_t i = 0; i < NPOINTS; ++i)
	{
		BitVector64 r((uint64_t)random());
		points.push_back(r);
	}

	slsh = new slash::SLSH<BitVector64>(d, k, L);
	lsh = new slash::LSH<BitVector64, slash::SLSH<BitVector64> >(d, k, L, slsh);
}

void cleanup()
{
	delete slsh;
	delete lsh;
}

void TestLinearSearch()
{
	std::cout << "==== " << __func__ << std::endl;

	BitVector64 &p = points[0];
	timespec start, end;
	slash::QueryContext<BitVector64> c(limit+1);

	clock_gettime(CLOCK_MONOTONIC, &start);
	for (size_t i = 0; i < NPOINTS; ++i)
	{
        BitVector64 &q = points[i];
        c.Insert(q, p.Similarity(q), q.NCopies());
	}
	clock_gettime(CLOCK_MONOTONIC, &end);
	const double del = 1e9 * (double)(end.tv_sec-start.tv_sec) + (double)(end.tv_nsec-start.tv_nsec);
	std::cout << del << " ns/op" << std::endl;

	c.shrink();
	auto neighbors = c.Neighbors();

	for (size_t i = 0; i < neighbors.size(); ++i)
	{
		std::cout << "n" << (int)i << ": " << p.Similarity(neighbors[i]) << ", " << neighbors[i].String(buf) << std::endl;
	}
}

void TestInsert()
{
	std::cout << "==== " << __func__ << std::endl;

	timespec start, end;

	clock_gettime(CLOCK_MONOTONIC, &start);

	lsh->Insert(points);

	clock_gettime(CLOCK_MONOTONIC, &end);
	const double del = 1e9 * (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec);
	std::cout << del << " ns/op" << std::endl;
}

void TestQuery()
{
	std::cout << "==== " << __func__ << std::endl;

	BitVector64 &p = points[0];
	size_t linearSearchSize = 0;

	printf("p: %s\n", p.String(buf));

	auto neighbors = lsh->Query(p, limit, &linearSearchSize);
	std::cout << "SLSH\nd=" << d << ", k=" << k <<", L=" << L << ", #points=" << (unsigned long long)NPOINTS << ", linearSearch=" << (double)linearSearchSize << std::endl;
	for (size_t i = 0; i < neighbors.size(); ++i)
	{
		std::cout <<"n" << (int)i << ": " << p.Similarity(neighbors[i]) << ", " << neighbors[i].String(buf) << std::endl;
	}
}

void BenchmarkQuery()
{
	std::cout << "==== " << __func__ << std::endl;

	timespec start, end;
	double del;
	double totalLinearSearchSize = 0;
	double totalFoundNeighbors = 0;
	size_t badLinearSearch = 0;
	size_t fewNeighbors = 0;

	clock_gettime(CLOCK_MONOTONIC, &start);
	for (size_t i = 0; i < NQUERIES; ++i)
	{
		size_t linearSearchSize = 0;

		auto neighbors = lsh->Query(points[i % (size_t)NPOINTS], limit, &linearSearchSize);

		totalLinearSearchSize += (double)linearSearchSize;
		if ((double)linearSearchSize > NPOINTS * BAD_LINEAR_SEARCH_FRACTION)
		{
			++badLinearSearch;
		}

		size_t size = neighbors.size();
		totalFoundNeighbors += (double)size;
		if (size < (size_t)limit / 3)
		{
			++fewNeighbors;
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &end);
	del = 1e9 * (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec);
	std::cout << (del / NQUERIES) <<" ns/op" << std::endl;
	std::cout << (totalLinearSearchSize / NQUERIES) << " linearly searched neighbors/op" << std::endl;
	std::cout << "average # of neighbors: " << (totalFoundNeighbors / NQUERIES) << std::endl;
	std::cout << "# of points with huge linear search (>" << ((double)NPOINTS * BAD_LINEAR_SEARCH_FRACTION) << " points): " << (double)badLinearSearch << " (" << (100.0 * (double)badLinearSearch / (double)NPOINTS) << "%%)" << std::endl;
	std::cout << "# of points with few neighbors (<" << (limit / 3) << "): " << (double)fewNeighbors << " (" << (100.0 * (double)fewNeighbors / (double)NPOINTS) << "%%)" << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_slash {

}  // namespace my_slash

int slash_main(int argc, char *argv[])
{
	local::init();

	local::TestLinearSearch();
	local::TestInsert();
	local::TestQuery();

	local::BenchmarkQuery();

    local::cleanup();

	return 0;
}
