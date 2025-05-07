#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_cgal {

void polygon();
void mesh();

}  // namespace my_cgal

int cgal_main(int argc, char *argv[])
{
	my_cgal::polygon();

	//my_cgal::mesh();  // Not yet implemented

	return 0;
}
