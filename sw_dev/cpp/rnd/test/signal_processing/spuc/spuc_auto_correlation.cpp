//#include "stdafx.h"
#include <spuc/auto_corr.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_spuc {

// [ref] ${SPUC_HOME}/examples/test_auto_corr.cpp
void auto_correlation_example()
{
    const int n = 15;

    SPUC::smart_array<double> a(n);
    SPUC::smart_array<double> b;

    //double z[n];
    //std::cout << "Size of z = " << sizeof(z) << std::endl;
    std::cout << "Size of a = " << sizeof(a) << std::endl;

    for (int i = 0; i < n; ++i)
        a[i] = -3.3 + (double)i;

    for (int i = 0; i < n; ++i)
        std::cout << a[i] << " ";
    std::cout << std::endl;

    b = SPUC::auto_corr(a);
    for (int i = 0; i < n; ++i)
        std::cout << b[i] << " ";
    std::cout << std::endl;
}

}  // namespace my_spuc
