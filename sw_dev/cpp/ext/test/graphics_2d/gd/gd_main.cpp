#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_gd {

void basic_operation();
void two_link_arm();

}  // namespace my_gd

int gd_main(int argc, char* argv[])
{
	//my_gd::basic_operation();

	//
	my_gd::two_link_arm();

	return 0;
}
