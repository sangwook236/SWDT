#include "../particle_filter_object_tracking_lib/defs.h"
#include "../particle_filter_object_tracking_lib/utils.h"
#include "../particle_filter_object_tracking_lib/particles.h"
#include "../particle_filter_object_tracking_lib/observation.h"
#include <iostream>
#include <string>
#include <vector>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_particle_filter_object_tracking {

// ${PARTICLE_FILTER_OBJECT_TRACKING_HOME}/src/observe.c
void observe_example()
{
	const std::string input_filename("./bayesian_filtering/");
	IplImage *in_img = cvLoadImage(input_filename.c_str(), 1);  // input image.
	if (!in_img)
	{
		std::cerr << "input file not found: " << input_filename << std::endl;
		return;
	}

	std::vector<std::string> ref_filenames;
	ref_filenames.push_back("./bayesian_filtering/");
	const int num_ref_imgs = (int)ref_filenames.size();  // count of player reference images.
	IplImage **ref_imgs = (IplImage **)malloc(num_ref_imgs * sizeof(IplImage *));  // array of player reference images.
	for (int i = 0; i < num_ref_imgs; ++i)
	{
		ref_imgs[i] = cvLoadImage(ref_filenames[i].c_str(), 1);
		if (!ref_imgs[i])
		{
			std::cerr << "reference file not found: " << ref_filenames[i] << std::endl;
			return;
		}
	}

	// compute HSV histogram over all reference image.
	IplImage *hsv_img = bgr2hsv(in_img);
	IplImage **hsv_ref_imgs = (IplImage **)malloc(num_ref_imgs * sizeof(IplImage *));
	histogram *ref_histo;
	for (int i = 0; i < num_ref_imgs; ++i)
		hsv_ref_imgs[i] = bgr2hsv(ref_imgs[i]);
	ref_histo = calc_histogram(hsv_ref_imgs, num_ref_imgs);
	normalize_histogram(ref_histo);

	// compute likelihood at every pixel in input image.
	std::cout << "Computing likelihood... " << std::endl;
	IplImage *l32f = likelihood_image(hsv_img, ref_imgs[0]->width, ref_imgs[0]->height, ref_histo);
	std::cout << "done" << std::endl;

	// convert likelihood image to uchar and display.
	double max;
	cvMinMaxLoc(l32f, NULL, &max, NULL, NULL, NULL);
	IplImage *l = cvCreateImage(cvGetSize(l32f), IPL_DEPTH_8U, 1);
	cvConvertScale(l32f, l, 255.0 / max, 0);
	
	cvNamedWindow("likelihood", 1);
	cvShowImage("likelihood", l);
	cvNamedWindow("image", 1);
	cvShowImage("image", in_img);

	cvWaitKey(0);
}

}  // namespace my_particle_filter_object_tracking
