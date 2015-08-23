#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_tapkee {

void minimal_cpp_example();
void RNA_cpp_example();
void precomputed_distance_cpp_example();

}  // namespace my_tapkee

int tapkee_main(int argc, char *argv[])
{
	my_tapkee::minimal_cpp_example();
	//my_tapkee::RNA_cpp_example();
	//my_tapkee::precomputed_distance_cpp_example();

	return 0;
}
