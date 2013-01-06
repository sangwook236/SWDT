#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace cvm {

void matrix_operation();
void matrix_function();
void vector_operation();
void vector_function();
void lu();
void cholesky();
void qr();
void eigen();
void svd();

}  // namespace cvm

int cvm_main(int argc, char *argv[])
{
	//cvm::matrix_operation();
	//cvm::matrix_function();
	//cvm::vector_operation();
	cvm::vector_function();

	//cvm::lu();
	//cvm::cholesky();
	//cvm::qr();
	//cvm::eigen();
	//cvm::svd();

	return 0;
}
