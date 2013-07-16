//include "stdafx.h"
// tracker
#include "ParticleFilter2D.h"
#include "ParticleFilter3D.h"
#include <Image/ByteImage.h>
#include <Image/ImageProcessor.h>
#include <Image/PrimitivesDrawer.h>
#include <Structs/Structs.h>
#include <Helpers/helpers.h>
#include <Interfaces/ApplicationHandlerInterface.h>
#include <Interfaces/MainWindowInterface.h>
#include <gui/GUIFactory.h>
#include <iostream>
#include <cstdlib>


namespace {
namespace local {

#define PROCESSING_FRAMERATE	30

// set this define if CParticleFilter3D is to be used (variable side length of the square)
//#define TRACK_3D

static void SimulateMovement(CByteImage *pImage)
{
	const int width = pImage->width;
	const int height = pImage->height;

	// clear image
	ImageProcessor::Zero(pImage);

	// do simulation of a movement
	static double x = 200;
	static double y = 150;
	static double k = 40;
	static double km = k + 15;
	static double dx = 0, dy = 0;
	static int n = 0;

	if (n++ % 5 == 0)
	{
		dx = (rand() % 1023) / 50.0 - 10;
		dy = (rand() % 1023) / 50.0 - 10;
	}

	x += dx;
	y += dy;

	if (x < km) x = km;
	if (x >= width - km) x = width - km - 1;
	if (y < km) y = km;
	if (y >= height - km) y = height - km - 1;

	int xx = int(x + 0.5);
	int yy = int(y + 0.5);
	int kk = int(k + 0.5);

	int i, j;

	// draw square, but draw only every second pixel in each direction
	MyRegion region;
	region.min_x = xx - kk;
	region.min_y = yy - kk;
	region.max_x = xx + kk;
	region.max_y = yy + kk;
	for (i = region.min_y; i < region.max_y; i += 2)
		for (j = region.min_x; j < region.max_x; j += 2)
			pImage->pixels[i * pImage->width + j] = 255;

	// add static grid
	for (i = 0; i < pImage->height; i += 20)
		for (j = 0; j < pImage->width; j++)
			pImage->pixels[i * pImage->width + j] = 255;

	for (i = 0; i < pImage->height; i++)
		for (j = 0; j < pImage->width; j += 20)
			pImage->pixels[i * pImage->width + j] = 255;

	// add noise
	for (i = 0; i < 40000; i++)
		pImage->pixels[(rand() % pImage->height) * pImage->width + (rand() % pImage->width)] = 255;
}

}  // namespace local
}  // unnamed namespace

namespace my_ivt {

// [ref] ${IVT_HOME}/examples/ParticleFilterDemo/src/main.cpp
void particle_filter_example()
{
	const int NUMBER_OF_PARTICLES = 200;
	const int width = 400;
	const int height = 300;

	CByteImage image(width, height, CByteImage::eGrayScale);
	CByteImage color(width, height, CByteImage::eRGB24);

	MyRegion region;
	int k = 40;

#ifdef TRACK_3D
	double result_configuration[DIMENSION_3D];
	CParticleFilter3D particle_filter(NUMBER_OF_PARTICLES, width, height, k);
#else
	double result_configuration[DIMENSION_2D];
	CParticleFilter2D particle_filter(NUMBER_OF_PARTICLES, width, height, k);
#endif

	// gui
	CApplicationHandlerInterface *pApplicationHandler = CreateApplicationHandler();
	pApplicationHandler->Reset();

	CMainWindowInterface *pMainWindow = CreateMainWindow(0, 0, width, height, "Particle Filter Demo");
	WIDGET_HANDLE pImageWidget = pMainWindow->AddImage(0, 0, width, height);

	pMainWindow->Show();

	for (int i = 0; i < 600; ++i)
	{
		unsigned int t = get_timer_value(true);

		local::SimulateMovement(&image);

		particle_filter.SetImage(&image);
		particle_filter.ParticleFilter(result_configuration);

#ifdef TRACK_3D
		k = int(result_configuration[2] + 0.5);
#endif

		region.min_x = int(result_configuration[0] - k + 0.5);
		region.min_y = int(result_configuration[1] - k + 0.5);
		region.max_x = int(result_configuration[0] + k + 0.5);
		region.max_y = int(result_configuration[1] + k + 0.5);

		ImageProcessor::ConvertImage(&image, &color);
		PrimitivesDrawer::DrawRegion(&color, region, 255, 0, 0, 2);

		pMainWindow->SetImage(pImageWidget, &color);

		if (pApplicationHandler->ProcessEventsAndGetExit())
			break;

		while (get_timer_value() - t < 1000000.0 / 30);
	}

	delete pMainWindow;
	delete pApplicationHandler;
}

}  // namespace my_ivt
