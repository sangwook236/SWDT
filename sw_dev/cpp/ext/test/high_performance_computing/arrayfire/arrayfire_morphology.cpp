/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

// REF [site] >> http://arrayfire.org/docs/image_processing_2morphing_8cpp-example.htm

#include <arrayfire.h>
#include <af/util.h>
//#include <af/opencl.h>
#include <iostream>
#include <string>
#include <chrono>


namespace {
namespace local {

af::array morphopen(const af::array& img, const af::array& mask)
{
	return af::dilate(af::erode(img, mask), mask);
}

af::array morphclose(const af::array& img, const af::array& mask)
{
	return af::erode(af::dilate(img, mask), mask);
}

af::array morphgrad(const af::array& img, const af::array& mask)
{
	return (af::dilate(img, mask) - af::erode(img, mask));
}

af::array tophat(const af::array& img, const af::array& mask)
{
	return (img - morphopen(img, mask));
}

af::array bottomhat(const af::array& img, const af::array& mask)
{
	return (morphclose(img, mask) - img);
}

af::array border(const af::array& img, const int left, const int right, const int top, const int bottom, const float value = 0.0)
{
	if ((int)img.dims(0) < (top + bottom))
		std::cout << "Input does not have enough rows." << std::endl;
	if ((int)img.dims(1) < (left + right))
		std::cerr << "Input does not have enough columns." << std::endl;

	af::dim4 imgDims = img.dims();
	af::array ret = af::constant(value, imgDims);
	ret(af::seq(top, imgDims[0] - bottom), af::seq(left, imgDims[1] - right), af::span, af::span) = img(af::seq(top, imgDims[0] - bottom), af::seq(left, imgDims[1] - right), af::span, af::span);

	return ret;
}

af::array border(const af::array& img, const int w, const int h, const float value = 0.0)
{
	return border(img, w, w, h, h, value);
}

af::array border(const af::array& img, const int size, const float value = 0.0)
{
	return border(img, size, size, size, size, value);
}

af::array blur(const af::array& img, const af::array mask = af::gaussianKernel(3,3))
{
	af::array blurred = af::array(img.dims(), img.type());
	for (int i = 0; i < (int)blurred.dims(2); ++i)
		blurred(af::span, af::span, i) = af::convolve(img(af::span, af::span, i), mask);
	return blurred;
}

void morphing_demo()
{
	af::Window wnd(1280, 720, "Morphological Operations");

	// Load images.
	const std::string image_filepath("../data/1180821115216L_3degree.tif");
	//const std::string image_filepath("../data/spl5_L14.png");
	af::array img_rgb = af::loadImage(image_filepath.c_str(), true) / 255.f;  // 3 channel RGB [0-1].

	af::array mask = af::constant(1, 5, 5);

	std::chrono::time_point<std::chrono::high_resolution_clock> startTime, endTime;
	startTime = std::chrono::high_resolution_clock::now();
	af::array er = af::erode(img_rgb, mask);
	endTime = std::chrono::high_resolution_clock::now();
	std::cout << "af::erode() took " << std::chrono::duration<double, std::milli>(endTime - startTime).count() << " ms." << std::endl;

	startTime = std::chrono::high_resolution_clock::now();
	af::array di = af::dilate(img_rgb, mask);
	endTime = std::chrono::high_resolution_clock::now();
	std::cout << "af::dilate() took " << std::chrono::duration<double, std::milli>(endTime - startTime).count() << " ms." << std::endl;

	startTime = std::chrono::high_resolution_clock::now();
	af::array op = morphopen(img_rgb, mask);
	endTime = std::chrono::high_resolution_clock::now();
	std::cout << "morphopen() took " << std::chrono::duration<double, std::milli>(endTime - startTime).count() << " ms." << std::endl;

	startTime = std::chrono::high_resolution_clock::now();
	af::array cl = morphclose(img_rgb, mask);
	endTime = std::chrono::high_resolution_clock::now();
	std::cout << "morphclose() took " << std::chrono::duration<double, std::milli>(endTime - startTime).count() << " ms." << std::endl;

	startTime = std::chrono::high_resolution_clock::now();
	af::array gr = morphgrad(img_rgb, mask);
	endTime = std::chrono::high_resolution_clock::now();
	std::cout << "morphgrad() took " << std::chrono::duration<double, std::milli>(endTime - startTime).count() << " ms." << std::endl;

	startTime = std::chrono::high_resolution_clock::now();
	af::array th = tophat(img_rgb, mask);
	endTime = std::chrono::high_resolution_clock::now();
	std::cout << "tophat() took " << std::chrono::duration<double, std::milli>(endTime - startTime).count() << " ms." << std::endl;

	startTime = std::chrono::high_resolution_clock::now();
	af::array bh = bottomhat(img_rgb, mask);
	endTime = std::chrono::high_resolution_clock::now();
	std::cout << "bottomhat() took " << std::chrono::duration<double, std::milli>(endTime - startTime).count() << " ms." << std::endl;

	startTime = std::chrono::high_resolution_clock::now();
	af::array bl = blur(img_rgb, af::gaussianKernel(5, 5));
	endTime = std::chrono::high_resolution_clock::now();
	std::cout << "blur() took " << std::chrono::duration<double, std::milli>(endTime - startTime).count() << " ms." << std::endl;

	startTime = std::chrono::high_resolution_clock::now();
	af::array bp = border(img_rgb, 20, 30, 40, 50, 0.5);
	endTime = std::chrono::high_resolution_clock::now();
	std::cout << "border(img_rgb, 20, 30, 40, 50, 0.5) took " << std::chrono::duration<double, std::milli>(endTime - startTime).count() << " ms." << std::endl;

	startTime = std::chrono::high_resolution_clock::now();
	af::array bo = border(img_rgb, 20);
	endTime = std::chrono::high_resolution_clock::now();
	std::cout << "border(img_rgb, 20) took " << std::chrono::duration<double, std::milli>(endTime - startTime).count() << " ms." << std::endl;

	while (!wnd.close())
	{
		wnd.grid(3, 4);

		wnd(0, 0).image(img_rgb, "Input");
		wnd(1, 0).image(er, "Erosion");
		wnd(2, 0).image(di, "Dilation");

		wnd(0, 1).image(op, "Opening");
		wnd(1, 1).image(cl, "Closing");
		wnd(2, 1).image(gr, "Gradient");

		wnd(0, 2).image(th, "TopHat");
		wnd(1, 2).image(bh, "BottomHat");
		wnd(2, 2).image(bl, "Blur");

		wnd(0, 3).image(bp, "Border to gray");
		wnd(1, 3).image(bo, "Border to black");

		wnd.show();
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_arrayfire {

void morphology()
{
	const int device = 0;

	//af::setBackend(AF_BACKEND_CPU);
	//af::setBackend(AF_BACKEND_CUDA);
	af::setBackend(AF_BACKEND_OPENCL);
	af::setDevice(device);
	af::info();

	std::cout << "** ArrayFire Image Morphing Demo **" << std::endl << std::endl;
	local::morphing_demo();
}

}  // namespace my_arrayfire
