//#include "stdafx.h"
#include <itpp/itbase.h>
#include <iostream>


namespace {
namespace local {

void vector_and_matrix_operation()
{
    itpp::vec a = itpp::linspace(0.0, 2.0, 2);
    itpp::vec b = "1.0 2.0";
    itpp::vec c = 2 * a + 3 * b;
    std::cout << "c =\n" << c << std::endl;

    itpp::mat A = "1.0 2.0; 3.0 4.0";
    itpp::mat B = "0.0 1.0; 1.0 0.0";
    itpp::mat C = A * B + 2 * A;
    std::cout << "C =\n" << C << std::endl;
    std::cout << "inverse of B =\n" << itpp::inv(B) << std::endl;
}

void IO_operation()
{
    {
        itpp::it_file ff;
        ff.open("./data/signal_processing/itpp/it_file_test.it");

        itpp::vec a = itpp::linspace(1, 20, 20);
        ff << itpp::Name("a") << a;

        ff.flush();
        ff.close();
    }

    {
        itpp::it_file ff;
        ff.open("./data/signal_processing/itpp/it_file_test.it");

        itpp::vec a;
        ff >> itpp::Name("a") >> a;

        std::cout << "a = " << a << std::endl;
    }
}

void timer_operation()
{
    itpp::Real_Timer tt;
    tt.tic();

    const long N = 1000000;
    long sum = 0;
    for (int i = 0; i < N; ++i)
        sum += i;

    tt.toc_print();

    std::cout << "The sum of all integers from 0 to " << (N - 1) << " equals " << sum << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_itpp {

}  // namespace my_itpp

int itpp_main(int argc, char *argv[])
{
    local::vector_and_matrix_operation();
    local::IO_operation();
    local::timer_operation();

	return 0;
}
