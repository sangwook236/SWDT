#include "gnuplot-iostream.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>


namespace my_gnuplot {

void pause_if_needed();

}  // namespace my_gnuplot

namespace {
namespace local {

// REF [file] >> ${GNUPLOT-IOSTREAM_HOME}/example-misc.cc
void basic_demo()
{
#if 1
	Gnuplot gp("D:/util/gnuplot/bin/gnuplot.exe -persist");
	//Gnuplot gp("gnuplot -persist");
	//Gnuplot gp;  // The enviroment variable GNUPLOT_IOSTREAM_CMD is used if set.
#elif 0
	// For debugging or manual editing of commands:
	Gnuplot gp(std::fopen("./data/visualization/gnuplot/plot.gp", "w"));
	//Gnuplot gp("tee ./data/visualization/gnuplot/plot.gp | gnuplot -persist");
	//Gnuplot gp("> plot.gp");
	//Gnuplot gp(stdout);
#endif

	std::vector<std::pair<double, double>> xy_pts_A;
	for (double x = -2; x < 2; x += 0.01)
	{
		const double y = x * x * x;
		xy_pts_A.push_back(std::make_pair(x, y));
	}

	std::vector<std::pair<double, double>> xy_pts_B;
	for (double alpha = 0; alpha < 1; alpha += 1.0 / 24.0)
	{
		const double theta = alpha * 2.0 * 3.14159;
		xy_pts_B.push_back(std::make_pair(std::cos(theta), std::sin(theta)));
	}

	gp << "set xrange [-2:2]\nset yrange [-2:2]\n";
	gp << "plot '-' with lines title 'cubic', '-' with points title 'circle'\n";
	gp.send1d(xy_pts_A);
	gp.send1d(xy_pts_B);

	my_gnuplot::pause_if_needed();
}

// REF [file] >> ${GNUPLOT-IOSTREAM_HOME}/example-misc.cc
void tmpfile_demo()
{
	Gnuplot gp("D:/util/gnuplot/bin/gnuplot.exe -persist");
	//Gnuplot gp("gnuplot -persist");
	//Gnuplot gp;  // The enviroment variable GNUPLOT_IOSTREAM_CMD is used if set.

	std::vector<std::pair<double, double>> xy_pts_A;
	for (double x = -2; x<2; x += 0.01)
	{
		const double y = x * x * x;
		xy_pts_A.push_back(std::make_pair(x, y));
	}

	std::vector<std::pair<double, double>> xy_pts_B;
	for (double alpha = 0; alpha<1; alpha += 1.0 / 24.0)
	{
		const double theta = alpha * 2.0 * 3.14159;
		xy_pts_B.push_back(std::make_pair(std::cos(theta), std::sin(theta)));
	}

	gp << "set xrange [-2:2]\nset yrange [-2:2]\n";

	// Data will be sent via a temporary file.
	// These are erased when you call gp.clearTmpfiles() or when gp goes out of scope.
	// If you pass a filename (i.e. "gp.file1d(pts, 'mydata.dat')"), then the named file will be created and won't be deleted.

	// Note: you need std::endl here in order to flush the buffer.
	// The send1d() function flushes automatically, but we're not using that here.
	gp << "plot" << gp.file1d(xy_pts_A) << "with lines title 'cubic',"
		<< gp.file1d(xy_pts_B) << "with points title 'circle'" << std::endl;

	my_gnuplot::pause_if_needed();
}

// REF [file] >> ${GNUPLOT-IOSTREAM_HOME}/example-misc.cc
void png_demo()
{
	Gnuplot gp("D:/util/gnuplot/bin/gnuplot.exe -persist");
	//Gnuplot gp("gnuplot -persist");
	//Gnuplot gp;  // The enviroment variable GNUPLOT_IOSTREAM_CMD is used if set.

	gp << "set terminal png\n";

	std::vector<double> y_pts;
	for (int i = 0; i < 1000; i++)
	{
		const double y = (i / 500.0 - 1) * (i / 500.0 - 1);
		y_pts.push_back(y);
	}

	std::cout << "Creating my_graph_1.png" << std::endl;
	gp << "set output './data/visualization/gnuplot/my_graph_1.png'\n";
	gp << "plot '-' with lines, sin(x/200) with lines\n";
	gp.send1d(y_pts);

	std::vector<std::pair<double, double>> xy_pts_A;
	for (double x = -2; x < 2; x += 0.01)
	{
		const double y = x * x * x;
		xy_pts_A.push_back(std::make_pair(x, y));
	}

	std::vector<std::pair<double, double>> xy_pts_B;
	for (double alpha = 0; alpha < 1; alpha += 1.0 / 24.0)
	{
		const double theta = alpha * 2.0 * 3.14159;
		xy_pts_B.push_back(std::make_pair(std::cos(theta), std::sin(theta)));
	}

	std::cout << "Creating my_graph_2.png" << std::endl;
	gp << "set output './data/visualization/gnuplot/my_graph_2.png'\n";
	gp << "set xrange [-2:2]\nset yrange [-2:2]\n";
	gp << "plot '-' with lines title 'cubic', '-' with points title 'circle'\n";
	gp.send1d(xy_pts_A);
	gp.send1d(xy_pts_B);
}

}  // namespace local
}  // unnamed namespace

namespace my_gnuplot {

// REF [site] >> http://www.stahlke.org/dan/gnuplot-iostream/
void gnuplot_iostream_example()
{
	local::basic_demo();
	local::tmpfile_demo();
	local::png_demo();
}

}  // namespace my_gnuplot
