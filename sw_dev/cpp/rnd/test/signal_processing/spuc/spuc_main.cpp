//#define GET_ROOTS 1

#include <itpp/itbase.h>
#include <spuc/vector.h>
#include <spuc/matrix.h>
#ifdef GET_ROOTS
#include <spuc/get_roots.h>
#endif
#include <spuc/find_roots.h>
#include <spuc/complex.h>
#include <spuc/spuc_typedefs.h>
#include <fstream>
#include <iostream>


namespace {
namespace local {

// [ref] ${SPUC_HOME}/examples/test_cmat.cpp
void matrix_operation_example()
{
    SPUC::matrix<SPUC::complex<double> > P(3, 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            P(i, j) = SPUC::complex<double>(std::rand() % 100, std::rand() % 100);
    //SPUC::vector<SPUC::complex<double> > k;
    //SPUC::vector<double> w, u, ut, x;
    SPUC::complex<double> d(3, 2);

    SPUC::matrix<SPUC::complex<double> > Z = d * P;
    SPUC::matrix<SPUC::complex<double> > Y = P * d;
    SPUC::matrix<SPUC::complex<double> > X = d + P;
}

// [ref] ${SPUC_HOME}/examples/test_ls_solve.cpp
void linear_system_example()
{
    const double M_2PI = std::atan(1) * 4.0 * 2.0;
    std::ofstream rf("./data/signal_processing/spuc/r.dat");

    const int M = 32;
    const int N = 4 * M;

    std::cout << "Real systems:" << std::endl << std::endl;

    itpp::mat A = itpp::randn(N, M);
    itpp::vec b = itpp::randn(N);

    // passband
    for (int i = 0; i < N; ++i)
    {
        double f = M_2PI * i / (2 * (N - 1));
        b(i) = 1.0;
        if (i >= N / 2) b(i) = 0;

        b(i) = 0;
        for (int n = 0; n < M; ++n)
        {
            A(i, n) = std::cos(n * f);
        }
    }

    std::cout << "Starting LS SOLVE..." << std::endl;
    const itpp::vec x = itpp::ls_solve_od(A, b);

    std::cout << "Square system: Ax = b" << std::endl
        << "============================" << std::endl
        << "A = " << A << std::endl
        << "b = " << b << std::endl
        << "x = " << x << std::endl << std::endl;

    for (int i = 1; i < M; ++i) rf << x(M - i) << "\n";
    rf << 2 * x(0) << "\n";
    for (int i = 1; i < M; ++i) rf << x(i) << "\n";

    rf.close();
}

// [ref] ${SPUC_HOME}/examples/test_ls_solve.cpp
void root_finding_example()
{
    //const std::size_t maxiter = 500;
    //const double DBL_EPSILON = 1e-20;
    const int n = 5;

    SPUC::float_type a[] = { 0.20882,  -0.97476,   2.08818,  -1.94952,   1.04409,  -0.19495 };
    SPUC::smart_array<SPUC::float_type> b(6);
    b[0] = -0.19495;
    b[1] =  1.04409;
    b[2] = -1.94952;
    b[3] =  2.08818;
    b[4] = -0.97476;
    b[5] =  0.20882;

    std::cout << "Polynomial order = " << n << std::endl;
    std::cout << "Enter coefficients, high order to low order" << std::endl;
    for (int i = 0; i <= n; ++i)
    {
        std::cout << a[i] << " * x^" << (n - i) << " : ";
    }
    std::cout << std::endl;

    // initialize estimate for 1st root pair
    SPUC::float_type quad[2];
    quad[0] = 2.71828e-1;
    quad[1] = 3.14159e-1;
    std::cout << quad[0] << " " << quad[1] << std::endl;

    // get p
#ifdef GET_ROOTS
    SPUC::get_quads(a, n, quad, x);
    int numr = SPUC::roots(x, n, wr);

    std::cout << "METHOD 1" << std::endl;

    std::cout << std::endl << "Roots (" << numr << " found):" << std::endl;
    std::cout.setf(std::ios::showpoint);
    std::cout.precision(15);
    std::cout.setf(std::ios::showpos);
    for (int i = 0; i < n; ++i)
    {
        std::cout.width(18);
        if ((std::real(wr[i]) != 0.0) || (std::imag(wr[i]) != 0.0))
          std::cout << wr[i].real() << " " << wr[i].imag() << "I" << std::endl;
    }
#endif

    std::cout << "METHOD 2" << std::endl;

    SPUC::smart_array<SPUC::complex<SPUC::float_type> > p = SPUC::find_roots(b, n);

    //std::cout << std::endl << "Roots (" << numr << " found):" << std::endl;
    std::cout.setf(std::ios::showpoint);
    std::cout.precision(15);
    std::cout.setf(std::ios::showpos);
    for (int i = 0; i < n; ++i)
    {
        std::cout.width(18);
        //std::cout << SPUC::real(p[i]) << " " << SPUC::imag(p[i]) << "I" << std::endl;
        std::cout << p[i].re << " " << p[i].im << "I" << std::endl;
    }
}

}  // namespace local
}  // unnamed namespace

namespace my_spuc {

void auto_correlation_example();
void filter_example();
void maximum_likelihood_sequence_estimation_example();

}  // namespace my_spuc

int spuc_main(int argc, char *argv[])
{
    //local::matrix_operation_example();
    //local::linear_system_example();
    //local::root_finding_example();

    my_spuc::auto_correlation_example();
    //my_spuc::filter_example();
    my_spuc::maximum_likelihood_sequence_estimation_example();

	return 0;
}
