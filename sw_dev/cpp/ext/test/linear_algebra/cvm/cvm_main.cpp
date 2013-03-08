#if defined(WIN32) || defined(_WIN32)
#include <cvm/cvm.h>
#else
#include <cvm.h>
#endif
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_cvm {

void matrix_operation();
void matrix_function();
void vector_operation();
void vector_function();
void lu();
void cholesky();
void qr();
void eigen();
void svd();

}  // namespace my_cvm

int cvm_main(int argc, char *argv[])
{
    try
    {
        //my_cvm::matrix_operation();
        //my_cvm::matrix_function();
        //my_cvm::vector_operation();
        my_cvm::vector_function();

        //my_cvm::lu();
        //my_cvm::cholesky();
        //my_cvm::qr();
        //my_cvm::eigen();
        //my_cvm::svd();
    }
	catch (const cvm::cvmexception &e)
	{
 		std::cout << "cvm::cvmexception caught: " << e.what() << std::endl;
	}

	return 0;
}
