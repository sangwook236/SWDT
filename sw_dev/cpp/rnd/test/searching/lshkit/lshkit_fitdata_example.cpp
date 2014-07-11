//#include "stdafx.h"
#include <lshkit.h>
#include <gsl/gsl_multifit.h>
#include <boost/progress.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <cmath>
#include <cstdlib>


namespace {
namespace local {

bool is_good_value(double v)
{
    return ((v > -std::numeric_limits<double>::max()) && (v < std::numeric_limits<double>::max()));
}

}  // namespace local
}  // unnamed namespace

namespace my_lshkit {

// ${LSHKIT_HOME}/tools/fitdata.cpp
void fitdata_example()
{
    const std::string data_file("./data/search_algorithm/lshkit/audio.data");
    const unsigned N = 0;  // number of points to use.
    const unsigned P = 50000;  // number of pairs to sample.
    unsigned Q = 1000;  // number of queries to sample.
    unsigned K = 100;  // search for K nearest neighbors.
    const unsigned F = 10;  // divide the sample to F folds.

    // load matrix.
    lshkit::Matrix<float> data(data_file);

    std::vector<unsigned> idx(data.getSize());
    for (unsigned i = 0; i < idx.size(); ++i) idx[i] = i;
    random_shuffle(idx.begin(), idx.end());

    if (N > 0 && N < data.getSize()) idx.resize(N);

    lshkit::metric::l2sqr<float> l2sqr(data.getDim());

    lshkit::DefaultRng rng;
    boost::variate_generator<lshkit::DefaultRng &, lshkit::UniformUnsigned> gen(rng, lshkit::UniformUnsigned(0, idx.size()-1));

    double gM = 0.0;
    double gG = 0.0;
    {
        // sample P pairs of points
        for (unsigned k = 0; k < P; ++k)
        {
            double dist, logdist;
            for (;;)
            {
                unsigned i = gen();
                unsigned j = gen();
                if (i == j) continue;
                dist = l2sqr(data[idx[i]], data[idx[j]]);
                logdist = std::log(dist);
                if (local::is_good_value(logdist)) break;
            }
            gM += dist;
            gG += logdist;
        }
        gM /= P;
        gG /= P;
        gG = std::exp(gG);
    }

    if (Q > idx.size()) Q = idx.size();
    if (K > idx.size() - Q) K = idx.size() - Q;
    // sample query.
    std::vector<unsigned> qry(Q);

    lshkit::SampleQueries(&qry, idx.size(), rng);

    // do the queries.
    std::vector<lshkit::Topk<unsigned> > topks(Q);
    for (unsigned i = 0; i < Q; ++i) topks[i].reset(K);

    /* ... */
    gsl_matrix *X = gsl_matrix_alloc(F * K, 3);
    gsl_vector *yM = gsl_vector_alloc(F * K);
    gsl_vector *yG = gsl_vector_alloc(F * K);
    gsl_vector *pM = gsl_vector_alloc(3);
    gsl_vector *pG = gsl_vector_alloc(3);
    gsl_matrix *cov = gsl_matrix_alloc(3,3);

    std::vector<double> M(K);
    std::vector<double> G(K);

    boost::progress_display progress(F, std::cerr);
    unsigned m = 0;
    for (unsigned l = 0; l < F; l++)
    {
        // Scan
        for (unsigned i = l; i< idx.size(); i += F)
        {
            for (unsigned j = 0; j < Q; j++)
            {
                int id = qry[j];
                if (i != id)
                {
                    float d = l2sqr(data[idx[id]], data[idx[i]]);
                    if (local::is_good_value(std::log(double(d)))) topks[j] << lshkit::Topk<unsigned>::Element(i, d);
                }
            }
        }

        std::fill(M.begin(), M.end(), 0.0);
        std::fill(G.begin(), G.end(), 0.0);

        for (unsigned i = 0; i < Q; i++)
        {
            for (unsigned k = 0; k < K; k++)
            {
                M[k] += topks[i][k].dist;
                G[k] += std::log(topks[i][k].dist);
            }
        }

        for (unsigned k = 0; k < K; k++)
        {
            M[k] = std::log(M[k]/Q);
            G[k] /= Q;
            gsl_matrix_set(X, m, 0, 1.0);
            gsl_matrix_set(X, m, 1, std::log(double(data.getSize() * (l + 1)) / double(F)));
            gsl_matrix_set(X, m, 2, std::log(double(k + 1)));
            gsl_vector_set(yM, m, M[k]);
            gsl_vector_set(yG, m, G[k]);
            ++m;
        }

        ++progress;
    }

    gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc(F * K, 3);

    double chisq;

    gsl_multifit_linear(X, yM, pM, cov, &chisq, work);
    gsl_multifit_linear(X, yG, pG, cov, &chisq, work);

    std::cout << gM << '\t' << gG << std::endl;
    std::cout << gsl_vector_get(pM, 0) << '\t'
         << gsl_vector_get(pM, 1) << '\t'
         << gsl_vector_get(pM, 2) << std::endl;
    std::cout << gsl_vector_get(pG, 0) << '\t'
         << gsl_vector_get(pG, 1) << '\t'
         << gsl_vector_get(pG, 2) << std::endl;

    gsl_matrix_free(X);
    gsl_matrix_free(cov);
    gsl_vector_free(pM);
    gsl_vector_free(pG);
    gsl_vector_free(yM);
    gsl_vector_free(yG);
}

}  // namespace my_lshkit
