#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_plplot {

void example_x01(int argc, const char **argv);
void example_x21(int argc, const char **argv);

}  // namespace my_plplot

int plplot_main(int argc, char *argv[])
{
	// examples
	my_plplot::example_x01(argc, (const char **)argv);  // run-time error
	my_plplot::example_x21(argc, (const char **)argv);  // run-time error

	return 0;
}

