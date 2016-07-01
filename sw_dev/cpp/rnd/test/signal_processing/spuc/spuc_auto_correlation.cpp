//#include "stdafx.h"
#include <spuc/dsp_functions/auto_corr.h>
#include <iostream>
#include <vector>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_spuc {

// REF [file] >> ${SPUC_HOME}/examples/test_auto_corr.cpp
void auto_correlation_example()
{
    const int n = 15;

    std::vector<double> a(n);
	std::vector<double> b;

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
