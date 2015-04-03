//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_optpp {

void example1();
void example2();

}  // namespace my_optpp

int nlopt_main(int argc, char *argv[])
{
	my_optpp::example1();
	//my_optpp::example2();

    return 0;
}
