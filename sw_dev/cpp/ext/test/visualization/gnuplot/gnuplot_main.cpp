#include <iostream>
#include <list>
#include <string>
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

}  // namespace local
}  // unnamed namespace

namespace my_gnuplot {

void pause_if_needed()
{
#if defined(_WIN64) || defined(_WIN32)
	// For Windows, prompt for a keystroke before the Gnuplot object goes out of scope so that the gnuplot window doesn't get closed.
	std::cout << "Press ENTER to continue..." << std::endl;
	//std::cin.clear();
	//std::cin.ignore(std::cin.rdbuf()->in_avail());
	std::cin.get();
#endif
}

void gnuplot_iostream_example();
void gnuplot_cpp_example();

}  // namespace my_gnuplot

int gnuplot_main(int argc, char *argv[])
{
	//local::pipe_example();

	my_gnuplot::gnuplot_iostream_example();
	//my_gnuplot::gnuplot_cpp_example();

	return 0;
}
