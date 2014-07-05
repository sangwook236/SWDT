//#include "stdafx.h"
#include <spuc/fir.h>
#include <spuc/fir_coeff.h>
#include <spuc/remez_fir.h>
#include <spuc/gaussian_fir.h>
#include <spuc/butterworth_fir.h>
#include <spuc/create_remez_lpfir.h>
#include <spuc/butterworth_iir.h>
#include <spuc/elliptic_iir.h>
#include <spuc/iir.h>
#include <spuc/iir_2nd.h>
#include <spuc/iir_allpass1.h>
#include <spuc/iir_allpass1_cascade.h>
#include <spuc/spuc_fp.h>
#include <spuc/freqz.h>
#include <spuc/spuc_typedefs.h>
#include <fstream>
#include <iostream>


namespace {
namespace local {

// [ref] ${SPUC_HOME}/examples/test_filters.cpp
void filters_example()
{
    const std::size_t PTS = 200;
    const std::size_t TAPS = 36;

    SPUC::float_type pass_edge = 0.245;
    SPUC::float_type stop_edge = 0.30625;

    SPUC::fir_coeff<SPUC::float_type> BF(TAPS);
    SPUC::fir_coeff<SPUC::float_type> RF(TAPS);
    SPUC::fir_coeff<SPUC::float_type> GF(TAPS);
    //SPUC::allpass_iir<SPUC::float_type> AF(0.23, 0.769, 2);
    SPUC::iir_allpass1_cascade<SPUC::float_type> AF1(0.4, 2);
    SPUC::iir_allpass1_cascade<SPUC::float_type> AF2(0.44, 3);

    SPUC::complex<SPUC::float_type> h[PTS];

    SPUC::create_remez_lpfir(RF, pass_edge, stop_edge, 1);
    SPUC::butterworth_fir(BF, 0.15);
    SPUC::gaussian_fir(GF, 0.25, 8);

    RF.print();
    BF.print();
    GF.print();

    //
    std::ofstream IMP("./data/signal_processing/spuc/impulses_filters.dat");
    SPUC::float_type imp = 1;
    for (int i = 0; i < 40; ++i)
     {
        IMP	<< AF1.clock(imp) << " "
            << AF2.clock(imp) << " "
            //<< CPF.clock(imp) << " "
            //<< EPF.clock(imp) << " "
            //<< BPF.clock(imp) << " "
            //<< BF.clock(imp) << " "
            //<< RF.clock(imp) << " "
            //<< GF.clock(imp) << " "
            << std::endl;
        imp = 0;
    }
    IMP.close();

    //
    std::ofstream HF("./data/signal_processing/spuc/h.dat");
    for (std::size_t i = 0; i < PTS; ++i)
        HF << (20.0 * std::log10(SPUC::magsq(h[i]))) << std::endl;

    HF.close();
}

// [ref] ${SPUC_HOME}/examples/test_fir2.cpp
void fir2_example()
{
    //SPUC::float_type x;
    //SPUC::float_type y;
    //SPUC::float_type pass_edge = 0.08;
    //SPUC::float_type stop_edge = 0.16;

    std::ofstream IMP("./data/signal_processing/spuc/impulses_fir2.dat");

    SPUC::fir<long, SPUC::float_type> test1(3);
    SPUC::fir<SPUC::complex<long>, SPUC::float_type> test2(3);
    SPUC::fir<SPUC::float_type, SPUC::float_type> test3(3);
    SPUC::fir<SPUC::complex<SPUC::float_type>, SPUC::float_type> test4(3);
    SPUC::fir<long, long> test5(3);
    SPUC::fir<SPUC::complex<long>, long> test6(3);
    SPUC::fir<SPUC::float_type, long> test7(3);
    SPUC::fir<SPUC::complex<SPUC::float_type>, long> test8(3);

    //test2.print();
    //LPF.print();

    //SPUC::float_type imp = 1;
    long iimp = 1;
    for (int i = 0; i < 100; ++i)
    {
        IMP << test1.clock(iimp) << std::endl;
        IMP << test2.clock(iimp) << std::endl;
        IMP << test3.clock(iimp) << std::endl;
        IMP << test4.clock(iimp) << std::endl;
        IMP << test5.clock(iimp) << std::endl;
        IMP << test6.clock(iimp) << std::endl;
        IMP << test7.clock(iimp) << std::endl;
        IMP << test8.clock(iimp) << std::endl;
        //imp = 0;
        iimp = 0;
    }

    IMP.close();
}

// [ref] ${SPUC_HOME}/examples/test_iir.cpp
void iir_example()
{
    const long N = 32;
    const long O = 4;
    //SPUC::float_type x;
    //SPUC::float_type y;
    //SPUC::float_type pass_edge = 0.2;
    //SPUC::float_type stop_edge = 0.22;

    SPUC::iir_coeff BPF(O);
    SPUC::butterworth_iir(BPF, 0.1, true, 3.0);
    SPUC::iir<SPUC::float_type> LPF(BPF);

    SPUC::iir<SPUC::spuc_int<16>, SPUC::float_type> IPF(BPF);
    SPUC::iir<SPUC::float_type, SPUC::spuc_fixed<30, 15> > SPF(BPF);

    std::ofstream IMP("./data/signal_processing/spuc/impulses_iir.dat");

    LPF.print();

    std::cout << "--------------------------------------" << std::endl;
    std::cout << "Int_s version" << std::endl;
    IPF.print();
    std::cout << "Int_s version end" << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    std::cout << "--------------------------------------" << std::endl;
    std::cout << "vfixed version" << std::endl;
    SPF.print();
    std::cout << "vfixed version end" << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    std::cout << "poles = ";
    for (int i = 0; i < (O+1)/2; ++i) std::cout << BPF.poles[i] << " ";
    std::cout << std::endl;
    std::cout << "zeros = ";
    for (int i = 0; i < (O+1)/2; ++i) std::cout << BPF.zeros[i] << " ";
    std::cout << std::endl;
    std::cout << "A = ";
    for (int i = 0; i < O+1; ++i) std::cout << BPF.a_tf[i] << " ";
    std::cout << std::endl;
    std::cout << "B = ";
    for (int i = 0; i < O+1; ++i) std::cout << BPF.b_tf[i] << " ";
    std::cout << std::endl;

    SPUC::float_type imp = 100.0;

    //SPUC::sint<16> iimp = SPUC::sint<16>((long)imp);
    //SPUC::vfixed fimp = SPUC::vfixed(imp);

    for (int i = 0; i < N; ++i)
    {
        std::cout << LPF.clock(imp) << " "
            //<< IPF.clock(imp) << " "
            << SPF.clock(imp) << " " << std::endl;
        imp = 0;
    }
}

// [ref] ${SPUC_HOME}/examples/test_iir2nd.cpp
void iir2nd_example()
{
    //double x;
    //double y;
    //double pass_edge = 0.08;
    //double stop_edge = 0.16;

    SPUC::iir_coeff BPF(2);
    SPUC::elliptic_iir(BPF, 0.2, true, 0.22, 40, 0.5);
    SPUC::butterworth_iir(BPF, 0.35, true, 3);
    SPUC::iir<double> LPF(BPF);
    LPF.print();

    const double dA1 = BPF.get_coeff_a(0);
    const double dA2 = BPF.get_coeff_a(1);

    const int ROUND_BITS = 8;

    const long A1 = (long)std::floor((1 << (ROUND_BITS - 1)) * dA1 + 0.5);
    const long A2 = (long)std::floor((1 << (ROUND_BITS - 1)) * dA2 + 0.5);

    std::ofstream IMP("./data/signal_processing/spuc/impulses_iir2nd.dat");

    // REAL TYPEs
    SPUC::iir_2nd<long, double> test1(dA1, dA2);
    SPUC::iir_2nd<double, double> test2(dA1, dA2);

    SPUC::iir_2nd<long, long> test3(1, 2, 1, A1, A2, ROUND_BITS);
    SPUC::iir_2nd<double, long> test4(1, 2, 1, A1, A2, ROUND_BITS);

    // COMPLEX TYPES
    SPUC::iir_2nd<SPUC::complex<long>, double> test5(dA1, dA2);
    SPUC::iir_2nd<SPUC::complex<double>, double> test6(dA1, dA2);

    SPUC::iir_2nd<SPUC::complex<long>, long> test7(1, 2, 1, A1, A2, ROUND_BITS);
    SPUC::iir_2nd<SPUC::complex<double>, long> test8(1, 2, 1, A1, A2, ROUND_BITS);

    test2.print();
    test3.print();

    //double imp = 1;
    long iimp = 100;
    for (int i = 0; i < 100; ++i)
    {
        //IMP << LPF.clock(imp) << std::endl;
        IMP << test1.clock(iimp) << " ";
        IMP << test2.clock(iimp) << "  ";
        IMP << test3.clock(iimp) << "  ";
        IMP << test4.clock(iimp) << "  ";
        IMP << test5.clock(iimp) << " ";
        IMP << test6.clock(iimp) << "  ";
        IMP << test7.clock(iimp) << "  ";
        IMP << test8.clock(iimp) << "  " << std::endl;
        //imp = 0;
        iimp = 0;
    }

    IMP.close();
}

}  // namespace local
}  // unnamed namespace

namespace my_spuc {

void filter_example()
{
    local::filters_example();
    local::fir2_example();
    local::iir_example();
    local::iir2nd_example();
}

}  // namespace my_spuc
