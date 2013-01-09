#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_lapack {

void cblas();
void clapack();

}  // namespace my_lapack

int lapack_main(int argc, char* argv[])
{
	my_lapack::cblas();
	my_lapack::clapack();

	return 0;
}
