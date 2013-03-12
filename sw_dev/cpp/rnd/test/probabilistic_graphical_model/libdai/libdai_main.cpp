#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_libdai {

void varset_example();
void sprinkler_example();
void sprinkler_gibbs_example();
void sprinkler_em_example();

}  // namespace my_libdai

int libdai_main(int argc, char *argv[])
{
	my_libdai::varset_example();
	my_libdai::sprinkler_example();
	my_libdai::sprinkler_gibbs_example();
	my_libdai::sprinkler_em_example();

	return 0;
}
