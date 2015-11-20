#include <coin/ClpSimplex.hpp>
#include <iostream>
#include <cassert>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_coin_or {

// REF [site] >> http://www.coin-or.org/Clp/userguide/ch02s02.html#id4768485
void clp_simple_example()
{
	ClpSimplex model;

	const int status = model.readMps("./data/optimization/p0033.mps");
	if (!status)
	{
		model.primal();
	}
}

}  // namespace my_coin_or
