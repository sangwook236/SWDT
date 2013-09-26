#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_plplot {

void x01_example(int argc, const char **argv);
void x21_example(int argc, const char **argv);

}  // namespace my_plplot

int plplot_main(int argc, char *argv[])
{
	// caution: for execution, driver files have to be loaded.

	// examples
	//my_plplot::x01_example(argc, (const char **)argv);
	my_plplot::x21_example(argc, (const char **)argv);

	return 0;
}

