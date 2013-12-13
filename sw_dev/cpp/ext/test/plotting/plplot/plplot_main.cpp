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
	// [NOTICE] caution >> for execution, driver files have to be loaded.
	//	set PLPLOT_DRV_DIR=${PLPLOT_INSTALL}/lib/plplot5.9.9/driversd
	//	set PLPLOT_LIB=${PLPLOT_INSTALL}\share\plplot5.9.9
	//	[ref] ${PLPLOT_HOME}/plplot_build_guide.txt.

	// examples
	my_plplot::x01_example(argc, (const char **)argv);
	//my_plplot::x21_example(argc, (const char **)argv);

	return 0;
}

