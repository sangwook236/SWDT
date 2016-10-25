#include "gnuplot_i.hpp"
#include <iostream>


namespace my_gnuplot {

void pause_if_needed();

}  // namespace my_gnuplot

namespace {
namespace local {

// REF [file] >> ${GNUPLOT-CPP_HOME}/example.cc
void command_example()
{
	Gnuplot gp;

	gp.cmd("set isosample 100\n");
	gp.cmd("min=-1\n");
	gp.cmd("max=1\n");
	//gp.cmd("pi=3.141592\n");
	gp.cmd("set hidden3d\n");
	gp.cmd("set pm3d\n");
	gp.cmd("set contour\n");
	gp.cmd("splot [min:max] [min:max] x*x+2*y*y-0.3*cos(3*pi*x)-0.4*cos(4*pi*y)+0.7\n");
	//gp.cmd("pause -1\n");

	my_gnuplot::pause_if_needed();
}

}  // namespace local
}  // unnamed namespace

namespace my_gnuplot {

// REF [site] >> https://github.com/orbitcowboy/gnuplot-cpp
void gnuplot_cpp_example()
{
	// If path-variable for gnuplot is not set, do it with:
#if defined(_WIN64) || defined(_WIN32)
	Gnuplot::set_GNUPlotPath("D:/util/gnuplot/bin", "gnuplot.exe");
#endif

#if defined(__APPLE__)
	// Set a special standard terminal for showonscreen (normally not needed),
	//	e.g. Mac users who want to use x11 instead of aqua terminal:
	Gnuplot::set_terminal_std("x11");
#endif 

	try
	{
		local::command_example();
	}
	catch (const GnuplotException &ex)
	{
		std::cout << "GnuplotException caught: " << ex.what() << std::endl;
	}
}

}  // namespace my_gnuplot
