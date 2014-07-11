//#include "stdafx.h"
#include <lshkit.h>
#include <boost/progress.hpp>
#include <boost/format.hpp>
#include <boost/timer.hpp>
#include <iostream>
#include <string>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_lshkit {

/*
 * This program randomly picks Q points from a dataset as queries, and
 * then linear-scan the database to find K-NN/R-NN for each query to produce
 * a benchmark file.  For each query, the query point itself is excluded
 * from the K-NN/R-NN list.
 *
 * You can specify both K and R and the prorgram will search for the K
 * points closest to queries which are within distance range of R.
 * If K = 0, then all points within distance range of R are returned.
 * The default value of R is the maximal value of float.
 */

// ${LSHKIT_HOME}/tools/scan.cpp
void scan_example()
{
    const std::string data_file("./data/search_algorithm/lshkit/audio.data");
    const std::string query_file("./data/search_algorithm/lshkit/audio_saved.query");

    const unsigned int K = 0;  // number of nearest neighbors.
    const unsigned int Q = 1;  // number of queries to sample.
    const unsigned int metric = 2;  // 1: L1; 2: L2.
    const unsigned int seed = 0;  // random number seed, 0 to use default.
    const float R = 10.0;  // distance range to search for.

    lshkit::Matrix<float> data(data_file);

    lshkit::Benchmark<unsigned> bench;
    bench.init(Q, data.getSize(), seed);
    boost::timer timer;

    timer.restart();
    if (1 == metric)
    {

        lshkit::metric::l1<float> l1(data.getDim());

        boost::progress_display progress(Q);
        for (unsigned i = 0; i < Q; ++i)
        {
            unsigned q = bench.getQuery(i);
            lshkit::Topk<unsigned> &topk = bench.getAnswer(i);
            topk.reset(K, R);
            for (unsigned j = 0; j < (unsigned)data.getSize(); ++j)
            {
                if (q == j) continue;
                topk << lshkit::Topk<unsigned>::Element(j, l1(data[q], data[j]));
            }
            ++progress;
        }
    }
    else if (2 == metric)
    {
        lshkit::metric::l2<float> l2(data.getDim());

        boost::progress_display progress(Q);
        for (unsigned i = 0; i < Q; ++i)
        {
            unsigned q = bench.getQuery(i);
            lshkit::Topk<unsigned> &topk = bench.getAnswer(i);
            topk.reset(K, R);
            for (unsigned j = 0; j < (unsigned)data.getSize(); ++j)
            {
                if (q == j) continue;
                topk << lshkit::Topk<unsigned>::Element(j, l2(data[q], data[j]));
            }
            ++progress;
        }
    }
    else
    {
        std::cout << "METRIC NOT SUPPORTED." << std::endl;
    }

    std::cout << boost::format("QUERY TIME: %1%s.") % timer.elapsed() << std::endl;
    bench.save(query_file);

    //timer.tuck("QUERY");
}

}  // namespace my_lshkit
