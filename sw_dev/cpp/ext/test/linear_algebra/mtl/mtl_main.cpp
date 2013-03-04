#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_mtl {

void vector_operation();
void matrix_operation();
void matrix_vector_operation();

}  // namespace my_mtl

int mtl_main(int argc, char *argv[])
{
	my_mtl::vector_operation();
	my_mtl::matrix_operation();
	my_mtl::matrix_vector_operation();

	return 0;
}
