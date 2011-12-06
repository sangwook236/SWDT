#include <gd/gd.h>
#include <boost/math/constants/constants.hpp>
#include <iostream>
#include <stdio.h>
#include <errno.h>


#if defined(__cplusplus)
extern "C" {
#endif

namespace {
namespace local {

void draw_link(gdImagePtr im, const int X, const int Y, const double L, const int W, const double theta, const int color)
{
	const int X2 = X + (int)std::floor(L * std::cos(theta) + 0.5), Y2 = Y + (int)std::floor(L * std::sin(theta) + 0.5);
	const int X_off = (int)std::floor(W * 0.5 * std::sin(theta) + 0.5);
	const int Y_off = (int)std::floor(W * 0.5 * std::cos(theta) + 0.5);

#if 0
	gdImageSetThickness(im, W);
	gdImageLine(im, X + Y_off, Y + X_off, X2 - Y_off, Y2 - X_off, color);
#else
	gdPoint points[4];
	points[0].x = X + X_off;
	points[0].y = Y - Y_off;
	points[1].x = X2 + X_off;
	points[1].y = Y2 - Y_off;
	points[2].x = X2 - X_off;
	points[2].y = Y2 + Y_off;
	points[3].x = X - X_off;
	points[3].y = Y + Y_off;
	gdImageFilledPolygon(im, points, 4, color);
#endif

#if 1
	// inscribe a filled ellipse in the image
	gdImageFilledEllipse(im, X, Y, W, W, color);
	gdImageFilledEllipse(im, X2, Y2, W, W, color);
#else
	gdImageFilledArc(im, X, Y, W, W, 0, 360, color, gdPie);
	gdImageFilledArc(im, X2, Y2, W, W, 0, 360, color, gdPie);
#endif
}

void draw_arm(const double theta1, const double theta2, const int index)
{
	const int WIDTH = 96, HEIGHT = 128;

	gdImagePtr im = gdImageCreate(WIDTH, HEIGHT);

	const int black = gdImageColorAllocate(im, 0, 0, 0);
	const int white = gdImageColorAllocate(im, 255, 255, 255);
	const int gray = gdImageColorAllocate(im, 127, 127, 127);

	//
	const int L1 = 30, W1 = 6;  // upper-arm
	const int L2 = 20, W2 = 6;  // forearm

	const int X1 = WIDTH / 3, Y1 = HEIGHT / 2;
	const int X2 = X1 + (int)std::floor(L1 * std::cos(theta1) + 0.5), Y2 = Y1 + (int)std::floor(L1 * std::sin(theta1) + 0.5);

	// draw arm
	draw_link(im, X1, Y1, L1, W1, theta1, white);  // upper-arm
	draw_link(im, X2, Y2, L2, W2, theta1 + theta2, white);  // forearm

	// open a file for writing. "wb" means "write binary", important under MSDOS, harmless under Unix
	std::ostringstream sstream;
	sstream << ".\\gd_data\\two_link_arm_" << index << ".png";
	std::FILE *pngout = std::fopen(sstream.str().c_str(), "wb");
	if (NULL == pngout)
	{
		if (EACCES == errno)
			std::cout << "error access" << std::endl;
		else if (EISDIR == errno)
			std::cout << "error directory" << std::endl;
		else if (ENOENT == errno)
			std::cout << "error file" << std::endl;

		return;
	}

	// output the image to the disk file in PNG format
	gdImagePng(im, pngout);

	// close the files
	std::fclose(pngout);

	// destroy the image in memory
	gdImageDestroy(im);
}

}  // namespace local
}  // unnamed namespace

#if defined(__cplusplus)
}
#endif

void two_link_arm()
{
	const double theta1 = boost::math::constants::pi<double>() / 4.0;
	const double theta2 = boost::math::constants::pi<double>() / 4.0;

	for (int i = 0; i < 5; ++i)
		for (int j = 0; j < 5; ++j)
			local::draw_arm(theta1 * (i - 2), theta2 * (j - 2), i * 5 + j + 1);
}
