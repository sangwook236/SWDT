#include <iostream>
#include <stdexcept>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_boom {

}  // namespace my_boom

int boom_main(int argc, char *argv[])
{
	throw std::runtime_error("Not yet implemented");

	return 0;
}
