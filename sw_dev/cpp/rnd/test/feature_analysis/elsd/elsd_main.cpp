#include "../elsd_lib/elsd.h"
#include "../elsd_lib/write_svg.h"
#include "../elsd_lib/valid_curve.h"
#include "../elsd_lib/process_curve.h"
#include "../elsd_lib/process_line.h"
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <iostream>
#include <string>
#include <cmath>


namespace {
namespace local {

// ${ELSD_HOME}/elsd.c
void elsd_example()
{
	void EllipseDetection(image_double image, double rho, double prec, double p, double eps, int smooth, int *ell_count, int *circ_count, int *line_count, char *fstr);
	image_double read_pgm_image_double(char * name);
	
	const std::string image_filename("./feature_analysis_data/stars.pgm");

	const double quant = 2.0;  // Bound to the quantization error on the gradient norm
	const double ang_th = 22.5;  // Gradient angle tolerance in degrees
	const double p = ang_th / 180.0;
	const double prec = M_PI * ang_th / 180.0;  // radian precision
	const double rho = quant / std::sin(prec);
	const double eps = 1.0;
	const int smooth = 1;

	image_double image = read_pgm_image_double((char *)image_filename.c_str());

	int ell_count = 0, line_count = 0, circ_count = 0;
	EllipseDetection(image, rho, prec, p, eps, smooth, &ell_count, &circ_count, &line_count, (char *)image_filename.c_str());

	std::cout << image_filename << std::endl;
	std::cout << ell_count << " elliptical arcs, " << circ_count << " circular arcs, " << line_count << " line segments" << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_elsd {

}  // namespace my_elsd

int elsd_main(int argc, char *argv[])
{
	try
	{
		// ellipse & line segment detector (ELSD) --------------------
		local::elsd_example();
	}
	catch (const cv::Exception &e)
	{
		//std::cout << "OpenCV exception caught: " << e.what() << std::endl;
		//std::cout << "OpenCV exception caught: " << cvErrorStr(e.code) << std::endl;
		std::cout << "OpenCV exception caught:" << std::endl
			<< "\tdescription: " << e.err << std::endl
			<< "\tline:        " << e.line << std::endl
			<< "\tfunction:    " << e.func << std::endl
			<< "\tfile:        " << e.file << std::endl;

		return 1;
	}

	return 0;
}
