#include "gnuplot_i.hpp"
#include <iostream>
#include <string>
#include <list>
#include <cstdio>


namespace {
namespace local {

void pipe_example()
{
#if defined(_WIN64) || defined(_WIN32)
	const std::string gnuplotPath("D:/util/gnuplot/bin/gnuplot.exe");
#else
	const std::string gnuplotPath("gnuplot");
#endif

#if defined(_WIN64) || defined(_WIN32)
	FILE *gnuplotPipe = _popen((gnuplotPath + " -persistent").c_str(), "w");
#else
	FILE *gnuplotPipe = popen((gnuplotPath + " -persistent").c_str(), "w");
#endif

	std::list<std::string> commands;
	commands.push_back("set title \"Data plot\"");
	commands.push_back("plot [-pi/2:pi] cos(x), -(sin(x) > sin(x+1) ? sin(x) : sin(x+1))");

	for (const auto &cmd : commands)
		fprintf(gnuplotPipe, "%s\n", cmd.c_str());

#if defined(_WIN64) || defined(_WIN32)
	_pclose(gnuplotPipe);
#else
	pclose(gnuplotPipe);
#endif
}

// REF [site] >> https://github.com/orbitcowboy/gnuplot-cpp
// REF [file] >> ${GNUPLOT-CPP_HOME}/example.cc
void gnuplot_cpp_example()
{
	// If path-variable for gnuplot is not set, do it with:
	Gnuplot::set_GNUPlotPath("D:/util/gnuplot/bin");

#if defined(__APPLE__)
	// Set a special standard terminal for showonscreen (normally not needed),
	//	e.g. Mac users who want to use x11 instead of aqua terminal:
	Gnuplot::set_terminal_std("x11");
#endif 

	try
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

		//
		std::cout << "Press ENTER to continue..." << std::endl;
		std::cin.clear();
		std::cin.ignore(std::cin.rdbuf()->in_avail());
		std::cin.get();
	}
	catch (const GnuplotException &ex)
	{
		std::cout << "GnuplotException caught: " << ex.what() << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_gnuplot {

}  // namespace my_gnuplot

int gnuplot_main(int argc, char *argv[])
{
	local::pipe_example();
	local::gnuplot_cpp_example();

	return 0;
}

