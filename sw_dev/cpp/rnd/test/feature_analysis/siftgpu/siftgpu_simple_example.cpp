//#include "stdafx.h"
#include <cstdlib>
#include <siftgpu/SiftGPU.h>
#include <vector>
#include <iostream>

#ifdef _DEBUG
#define IL_DEBUG
#endif  // _DEBUG

#define ILUT_USE_OPENGL
#if defined(WIN32)
#include <IL/config.h>
#endif
#include <IL/il.h>
#include <IL/ilu.h>
#include <IL/ilut.h>

#ifdef __cplusplus
#include <IL/devil_cpp_wrapper.hpp>
#endif


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_siftgpu {

// [ref] ${SIFTGPU_HOME}/src/TestWin/SimpleSIFT.cpp
void simple_example_1()
{
	// processing parameters first
	//	-fo -1, starting from -1 octave
	//	-v 1, only print out # feature and overall time
	const char *my_argv[] ={ "-fo", "-1", "-v", "1" };

	// create a SiftGPU instance
	SiftGPU sift;
	sift.ParseParam(4, const_cast<char **>(my_argv));

	// create an OpenGL context for computation
	const int support = sift.CreateContextGL();
	// call VerfifyContexGL instead if using your own GL context
	//const int support = sift.VerifyContextGL();

	if (SiftGPU::SIFTGPU_FULL_SUPPORTED != support)
	{
		std::cout << "SiftGPU not fully supported" << std::endl;
		return;
	}

	//
	{
		// process an image, and save ASCII format SIFT files
		if (sift.RunSIFT("./data/feature_analysis/sift/640-1.jpg"))
			sift.SaveSIFT("./data/feature_analysis/sift/640-1.sift");
		else
			std::cout << "SIFT features not found" << std::endl;

		// you can get the feature vector and store it yourself
		if (!sift.RunSIFT("./data/feature_analysis/sift/640-2.jpg"))
		{
			std::cout << "SIFT features not found" << std::endl;
			return;
		}

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
		const char * files[4] = { "./data/feature_analysis/sift/640-1.jpg", "./data/feature_analysis/sift/640-2.jpg", "./data/feature_analysis/sift/640-3.jpg", "./data/feature_analysis/sift/640-4.jpg" };
		sift.SetImageList(4, files);

		// now you can process an image with its index
		sift.RunSIFT(1);
		sift.RunSIFT(0);
	}
}

void simple_example_2()
{
#ifdef SIFTGPU_DLL_RUNTIME

#	ifdef _WIN32
#		ifdef _DEBUG
    HMODULE hsiftgpu = LoadLibrary("siftgpu_d.dll");
#		else
    HMODULE hsiftgpu = LoadLibrary("siftgpu.dll");
#		endif
#	else
    void *hsiftgpu = dlopen("libsiftgpu.so", RTLD_LAZY);
#	endif

	if (NULL == hsiftgpu)
	{
		std::cout << "SiftGPU module not loaded" << std::endl;
		return;
	}

#	ifdef REMOTE_SIFTGPU
    ComboSiftGPU * (*pCreateRemoteSiftGPU)(int, char *) = NULL;
    pCreateRemoteSiftGPU = (ComboSiftGPU * (*)(int, char *))GET_MYPROC(hsiftgpu, "CreateRemoteSiftGPU");
    ComboSiftGPU *combo = pCreateRemoteSiftGPU(REMOTE_SERVER_PORT, REMOTE_SERVER);
    SiftGPU *sift = combo;
    SiftMatchGPU *matcher = combo;
#	else
    SiftGPU * (*pCreateNewSiftGPU)(int) = NULL;
    SiftMatchGPU * (*pCreateNewSiftMatchGPU)(int) = NULL;
    pCreateNewSiftGPU = (SiftGPU * (*)(int))GET_MYPROC(hsiftgpu, "CreateNewSiftGPU");
    pCreateNewSiftMatchGPU = (SiftMatchGPU * (*)(int))GET_MYPROC(hsiftgpu, "CreateNewSiftMatchGPU");
    SiftGPU *sift = pCreateNewSiftGPU(1);
    SiftMatchGPU *matcher = pCreateNewSiftMatchGPU(4096);
#	endif

#elif defined(REMOTE_SIFTGPU)
    ComboSiftGPU *combo = CreateRemoteSiftGPU(REMOTE_SERVER_PORT, REMOTE_SERVER);
    SiftGPU *sift = combo;
    SiftMatchGPU *matcher = combo;
#else
    // this will use overloaded new operators
    SiftGPU *sift = new SiftGPU;
    SiftMatchGPU *matcher = new SiftMatchGPU(4096);
#endif

    std::vector<float> descriptors1(1), descriptors2(1);
    std::vector<SiftGPU::SiftKeypoint> keys1(1), keys2(1);
    int num1 = 0, num2 = 0;

	if (ilGetInteger(IL_VERSION_NUM) < IL_VERSION ||
		iluGetInteger(ILU_VERSION_NUM) < ILU_VERSION ||
		ilutGetInteger(ILUT_VERSION_NUM) < ILUT_VERSION)
	{
		std::cout << "DevIL library is out of date! Please upgrade" << std::endl;
		return;
	}

	// initialize IL
	ilInit();
	// initialize ILU
	iluInit();

    // process parameters
    //	The following parameters are default in V340
    //		-m,    up to 2 orientations for each feature (change to single orientation by using -m 1)
    //		-s     enable subpixel subscale (disable by using -s 0)

	const char *my_argv[] = {
		"-fo", "-1",
		"-v", "1"
	};
    // -fo -1    staring from -1 octave
    // -v 1      only print out # feature and overall time
    // -loweo    add a (.5, .5) offset
    // -tc <num> set a soft limit to number of detected features

    // NEW:  parameters for GPU-selection
    // 1. CUDA.   Use parameter "-cuda", "[device_id]"
    // 2. OpenGL. Use "-Display", "display_name" to select monitor/GPU (XLIB/GLUT) on windows the display name would be something like \\.\DISPLAY4

    //////////////////////////////////////////////////////////////////////////////////////
    // You use CUDA for nVidia graphic cards by specifying
    // -cuda : cuda implementation (fastest for smaller images)
    //         CUDA-implementation allows you to create multiple instances for multiple threads
	//         Checkout src\TestWin\MultiThreadSIFT
    /////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////Two Important Parameters///////////////////////////
    // First, texture reallocation happens when image size increases, and too many
    // reallocation may lead to allocatoin failure.  You should be careful when using
    // siftgpu on a set of images with VARYING imag sizes. It is recommended that you
    // preset the allocation size to the largest width and largest height by using function
    // AllocationPyramid or prameter '-p' (e.g. "-p", "1024x768").

    // Second, there is a parameter you may not be aware of: the allowed maximum working
    // dimension. All the SIFT octaves that needs a larger texture size will be skipped.
    // The default prameter is 2560 for the unpacked implementation and 3200 for the packed.
    // Those two default parameter is tuned to for 768MB of graphic memory. You should adjust
    // it for your own GPU memory. You can also use this to keep/skip the small featuers.
    // To change this, call function SetMaxDimension or use parameter "-maxd".
	//
	// NEW: by default SiftGPU will try to fit the cap of GPU memory, and reduce the working
	// dimension so as to not allocate too much. This feature can be disabled by -nomc
    //////////////////////////////////////////////////////////////////////////////////////


    const int my_argc = sizeof(my_argv) / sizeof(char *);
    sift->ParseParam(my_argc, (char **)my_argv);

    ///////////////////////////////////////////////////////////////////////
    // Only the following parameters can be changed after initialization (by calling ParseParam).
    // -dw, -ofix, -ofix-not, -fo, -unn, -maxd, -b
    // to change other parameters at runtime, you need to first unload the dynamically loaded libaray
    // reload the libarary, then create a new siftgpu instance


    // Create a context for computation, and SiftGPU will be initialized automatically
    // The same context can be used by SiftMatchGPU
    if (sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED)
	{
		std::cout << "SiftGPU not fully supported" << std::endl;
		return;
	}

    if (sift->RunSIFT("./data/feature_analysis/sift/800-1.jpg"))
    {
        // Call SaveSIFT to save result to file, the format is the same as Lowe's
        sift->SaveSIFT("./data/feature_analysis/sift/800-1.sift");  // Note that saving ASCII format is slow

        // get feature count
        num1 = sift->GetFeatureNum();

        // allocate memory
        keys1.resize(num1);
		descriptors1.resize(128 * num1);

        // reading back feature vectors is faster than writing files
        // if you dont need keys or descriptors, just put NULLs here
        sift->GetFeatureVector(&keys1[0], &descriptors1[0]);
        // this can be used to write your own sift file.
    }
    else
        std::cerr << "SIFT running error" << std::endl;

    // You can have at most one OpenGL-based SiftGPU (per process).
    // Normally, you should just create one, and reuse on all images.
    if (sift->RunSIFT("./data/feature_analysis/sift/640-1.jpg"))
    {
        num2 = sift->GetFeatureNum();
        keys2.resize(num2);
		descriptors2.resize(128 * num2);
        sift->GetFeatureVector(&keys2[0], &descriptors2[0]);
    }
    else
        std::cerr << "SIFT running error" << std::endl;

	if (0 == num1 || 0 == num2)
	{
		std::cerr << "SIFT features not found" << std::endl;
		return;
	}

    // Testing code to check how it works when image size varies
    //sift->RunSIFT("./data/feature_analysis/sift/256.jpg");
	//sift->SaveSIFT("./data/feature_analysis/sift/256.sift.1");
    //sift->RunSIFT("./data/feature_analysis/sift/1024.jpg");  // this will result in pyramid reallocation
    //sift->RunSIFT("./data/feature_analysis/sift/256.jpg");
	//sift->SaveSIFT("./data/feature_analysis/sift/256.sift.2");
    // two sets of features for 256.jpg may have different order due to implementation

    //*************************************************************************
    /////compute descriptors for user-specified keypoints (with or without orientations)

    // Method1, set new keypoints for the image you've just processed with siftgpu
    //vector<SiftGPU::SiftKeypoint> mykeys;
    //sift->RunSIFT(mykeys.size(), &mykeys[0]);
    //sift->RunSIFT(num2, &keys2[0], 1);
	//sift->SaveSIFT("./data/feature_analysis/sift/640-1.sift.2");
    //sift->RunSIFT(num2, &keys2[0], 0);
	//sift->SaveSIFT("./data/feature_analysis/sift/640-1.sift.3");

    // Method2, set keypoints for the next coming image
    // The difference of with method 1 is that method 1 skips gaussian filtering
    //SiftGPU::SiftKeypoint mykeys[100];
    //for (int i = 0; i < 100; ++i)
	//{
    //    mykeys[i].s = 1.0f;
	//    mykeys[i].o = 0.0f;
    //    mykeys[i].x = (i % 10) * 10.0f + 50.0f;
    //    mykeys[i].y = (i / 10) * 10.0f + 50.0f;
    //}
    //sift->SetKeypointList(100, mykeys, 0);
    //sift->RunSIFT("./data/feature_analysis/sift/800-1.jpg");
	//sift->SaveSIFT("./data/feature_analysis/sift/800-1.sift.2");
    //### for comparing with method1:
    //sift->RunSIFT("./data/feature_analysis/sift/800-1.jpg");
    //sift->RunSIFT(100, mykeys, 0);
	//sift->SaveSIFT("./data/feature_analysis/sift/800-1.sift.3");
    //*********************************************************************************


    //**********************GPU SIFT MATCHING*********************************
    //**************************select shader language*************************
    // SiftMatchGPU will use the same shader lanaguage as SiftGPU by default
    // Before initialization, you can choose between glsl, and CUDA(if compiled).
    //matcher->SetLanguage(SiftMatchGPU::SIFTMATCH_CUDA);  // +i for the (i+1)-th device

    // Verify current OpenGL Context and initialize the Matcher;
    // If you don't have an OpenGL Context, call matcher->CreateContextGL instead;
    matcher->VerifyContextGL();  // must call once

    // Set descriptors to match, the first argument must be either 0 or 1
    // if you want to use more than 4096 or less than 4096
    // call matcher->SetMaxSift() to change the limit before calling setdescriptor
    matcher->SetDescriptors(0, num1, &descriptors1[0]);  // image 1
    matcher->SetDescriptors(1, num2, &descriptors2[0]);  // image 2

    // match and get result.
    int (*match_buf)[2] = new int [num1][2];
    // use the default thresholds. Check the declaration in SiftGPU.h
    int num_match = matcher->GetSiftMatch(num1, match_buf);
    std::cout << num_match << " sift matches were found." << std::endl;

    // enumerate all the feature matches
    for (int i  = 0; i < num_match; ++i)
    {
        // How to get the feature matches:
        //SiftGPU::SiftKeypoint & key1 = keys1[match_buf[i][0]];
        //SiftGPU::SiftKeypoint & key2 = keys2[match_buf[i][1]];
        // key1 in the first image matches with key2 in the second image
    }

    //*****************GPU Guided SIFT MATCHING***************
    // example: define a homography, and use default threshold 32 to search in a 64x64 window
    //float h[3][3] = { {0.8f, 0.0f, 0.0f}, {0, 0.8f, 0.0f}, {0.0f, 0.0f, 1.0f} };
    //matcher->SetFeatureLocation(0, &keys1[0]);  // SetFeatureLocaiton after SetDescriptors
    //matcher->SetFeatureLocation(1, &keys2[0]);
    //num_match = matcher->GetGuidedSiftMatch(num1, match_buf, h, NULL);
    //std::cout << num_match << " guided sift matches were found." << std::endl;
    // if you can want to use a Fundamental matrix, check the function definition

    // clean up
    delete [] match_buf;

#ifdef REMOTE_SIFTGPU
    delete combo;
#else
    delete sift;
    delete matcher;
#endif

#ifdef SIFTGPU_DLL_RUNTIME
    FREE_MYLIB(hsiftgpu);
#endif
}

}  // namespace my_siftgpu

