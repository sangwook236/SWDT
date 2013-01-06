#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace lapack {

void cblas_main();
void clapack_main();

}  // namespace lapack

int lapack_main(int argc, char* argv[])
{
	lapack::cblas_main();
	lapack::clapack_main();

	return 0;
}
