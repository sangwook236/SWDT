//#include "stdafx.h"
#include <cstdlib>
#include <siftgpu/SiftGPU.h>
#include <vector>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace siftgpu {

}  // namespace siftgpu

int siftgpu_main(int argc, char *argv[])
{
	// processing parameters first
	//	-fo -1, starting from -1 octave
	//	-v 1, only print out # feature and overall time
	const char * my_argv[] ={ "-fo", "-1", "-v", "1" };

	// create a SiftGPU instance
	SiftGPU sift;
	//sift.ParseParam(4, const_cast<char **>(my_argv));

	// create an OpenGL context for computation
	const int support = sift.CreateContextGL();
	// call VerfifyContexGL instead if using your own GL context
	//const int support = sift.VerifyContextGL();

	if (SiftGPU::SIFTGPU_FULL_SUPPORTED != support) return -1;

	//
	{
		// process an image, and save ASCII format SIFT files
		if (sift.RunSIFT("..\\feature_analysis_data\\siftgpu\\640-1.jpg"))
			sift.SaveSIFT("..\\feature_analysis_data\\siftgpu\\640-1.sift");

		// you can get the feature vector and store it yourself
		sift.RunSIFT("..\\feature_analysis_data\\siftgpu\\640-2.jpg");

		const int num = sift.GetFeatureNum();  // get feature count
		// allocate memory for read back
		std::vector<float> descriptors(128 * num);
		std::vector<SiftGPU::SiftKeypoint> keys(num);

		// read back keypoints and normalized descritpros
		// specify NULL if you don't need keypionts or descriptors
		sift.GetFeatureVector(&keys[0], &descriptors[0]);
	}
	
	{
		//const int width = ¡¦, height =¡¦;
		//const unsigned char *data = ¡¦;  // your (intensity) image data
		//sift.RunSIFT(width, height, data, GL_RGBA, GL_UNSIGNED_BYTE);
		// using GL_LUMINANCE data saves transfer time
	}

	{
		const char * files[4] = { "..\\feature_analysis_data\\siftgpu\\640-1.jpg", "..\\feature_analysis_data\\siftgpu\\640-2.jpg", "..\\feature_analysis_data\\siftgpu\\640-3.jpg", "..\\feature_analysis_data\\siftgpu\\640-4.jpg" };
		sift.SetImageList(4, files);

		// now you can process an image with its index
		sift.RunSIFT(1);
		sift.RunSIFT(0);
	}

	return 0;
}
