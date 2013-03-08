#include <iostream>
#include <stdexcept>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_openhpe {

void example();

}  // namespace my_openhpe

int openhpe_main(int argc, char *argv[])
{
	my_openhpe::example();

	return 0;
}
