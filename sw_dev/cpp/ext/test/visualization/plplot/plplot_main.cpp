#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_plplot {

void x01_example(int argc, char *argv[]);
void x21_example(int argc, char *argv[]);

}  // namespace my_plplot

int plplot_main(int argc, char *argv[])
{
	// NOTICE [caution] >> For execution, driver files have to be loaded.
	//	set PLPLOT_DRV_DIR=${PLPLOT_INSTALL}/lib/plplot5.9.9/driversd
	//	set PLPLOT_LIB=${PLPLOT_INSTALL}/share/plplot5.9.9
	//	REF [file] >> ${PLPLOT_HOME}/plplot_build_guide.txt.

	// Examples.
	my_plplot::x01_example(argc, argv);
	//my_plplot::x21_example(argc, argv);

	return 0;
}

