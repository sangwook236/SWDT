#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_galib {

void ex1(int argc, char *argv[]);
void ex3(int argc, char *argv[]);
void ex7(int argc, char *argv[]);

}  // namespace my_galib

int galib_main(int argc, char *argv[])
{
	my_galib::ex1(argc, argv);
	my_galib::ex3(argc, argv);
	my_galib::ex7(argc, argv);

	return 0;
}

