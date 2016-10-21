#include <coin/ClpSimplex.hpp>
#include <iostream>


namespace {
namespace local {

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

}  // namespace local
}  // unnamed namespace

namespace my_coin_or {

void clp_example()
{
	local::clp_simple_example();
}

}  // namespace my_coin_or
