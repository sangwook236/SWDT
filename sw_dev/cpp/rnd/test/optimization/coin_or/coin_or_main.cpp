//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_coin_or {

void clp_example();
void cbc_example();

void osi_example();

}  // namespace my_coin_or

int coin_or_main(int argc, char *argv[])
{
	// CLP: COIN-OR Linear Programming Solver.
	//my_coin_or::clp_example();

	// CBC: COIN-OR Branch-and-Cut MIP Solver.
	//my_coin_or::cbc_example();

	// OSI: COIN-OR Open Solver Interface.
	my_coin_or::osi_example();

    return 0;
}
