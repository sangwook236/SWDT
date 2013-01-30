#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_thrust {

void basic_operation();

void vector();
void list();

void algorithm();
void iterator();

void dot_product();

}  // namespace my_thrust

int thrust_main(int argc, char *argv[])
{
	my_thrust::basic_operation();

	my_thrust::vector();
	my_thrust::list();

	my_thrust::algorithm();
	my_thrust::iterator();

	my_thrust::dot_product();

    return 0;
}
