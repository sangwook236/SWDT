#include "stdafx.h"
#include <cstdlib>
#include <siftgpu/SiftGPU.h>
#include <vector>
#include <iostream>


int main()
{
	// processing parameters first
	//	-fo -1, starting from -1 octave
	//	-v 1, only print out # feature and overall time
	const char * argv[] ={ "-fo", "-1", "-v", "1" };

	// create a SiftGPU instance
	SiftGPU sift;
	//sift.ParseParam(4, const_cast<char **>(argv));

	// create an OpenGL context for computation
	const int support = sift.CreateContextGL();
	// call VerfifyContexGL instead if using your own GL context
	//const int support = sift.VerifyContextGL();

	if (SiftGPU::SIFTGPU_FULL_SUPPORTED != support) return -1;

	//
	{
		// process an image, and save ASCII format SIFT files
		if (sift.RunSIFT("..\\siftgpu_data\\640-1.jpg"))
			sift.SaveSIFT("..\\siftgpu_data\\640-1.sift");

		// you can get the feature vector and store it yourself
		sift.RunSIFT("..\\siftgpu_data\\640-2.jpg");

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
		const char * files[4] = { "..\\siftgpu_data\\640-1.jpg", "..\\siftgpu_data\\640-2.jpg", "..\\siftgpu_data\\640-3.jpg", "..\\siftgpu_data\\640-4.jpg" };
		sift.SetImageList(4, files);

		// now you can process an image with its index
		sift.RunSIFT(1);
		sift.RunSIFT(0);
	}

	std::wcout << L"press any key to exit ..." << std::endl;
	std::wcin.get();
	return 0;
}
