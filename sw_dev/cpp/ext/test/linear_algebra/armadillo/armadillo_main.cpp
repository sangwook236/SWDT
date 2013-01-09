#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_armadillo {

void vector_arithmetic();
void matrix_arithmetic();
void cube_arithmetic();

}  // namespace my_armadillo

int armadillo_main(int argc, char* argv[])
{
	//my_armadillo::vector_arithmetic();
	//my_armadillo::matrix_arithmetic();
	my_armadillo::cube_arithmetic();

    return 0;
}

