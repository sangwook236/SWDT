#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace armadillo {

void vector_arithmetic();
void matrix_arithmetic();
void cube_arithmetic();

}  // namespace armadillo

int armadillo_main(int argc, char* argv[])
{
	//armadillo::vector_arithmetic();
	//armadillo::matrix_arithmetic();
	armadillo::cube_arithmetic();

    return 0;
}

