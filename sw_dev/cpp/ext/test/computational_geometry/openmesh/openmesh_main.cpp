#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_openmesh {

void simple_object();

}  // namespace my_openmesh

int openmesh_main(int argc, char *argv[])
{
	// tutorials
	my_openmesh::simple_object();

	return 0;
}
