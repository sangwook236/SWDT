//#include "stdafx.h"
#include <iostream>
#include <string>

#ifdef _DEBUG
#define IL_DEBUG
#endif  // _DEBUG

#define ILUT_USE_OPENGL
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
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

void load_and_save_in_c()
{
#if defined(UNICODE) || defined(_UNICODE)
	const std::wstring input_image_filename(L"./data/graphics_2d/devil/640-5.jpg");
	const std::wstring output_image_filename(input_image_filename + L"_c.png");
#else
	const std::string input_image_filename("./data/graphics_2d/devil/640-5.jpg");
	const std::string output_image_filename(input_image_filename + "_c.png");
#endif

	if (ilGetInteger(IL_VERSION_NUM) < IL_VERSION ||
		iluGetInteger(ILU_VERSION_NUM) < ILU_VERSION ||
		ilutGetInteger(ILUT_VERSION_NUM) < ILUT_VERSION)
	{
		std::cout << "DevIL library is out of date! Please upgrade" << std::endl;
		return;
	}

	// Initialize IL.
	ilInit();
	// Initialize ILU.
	iluInit();

	// Initialize ILUT with OpenGL support.
	//	ILUT_OPENGL – initializes ILUT's OpenGL support.
	//	ILUT_ALLEGRO – initializes ILUT's Allegro support.
	//	ILUT_WIN32 – initializes ILUT's Windows GDI and DirectX 8 support.
	ilutRenderer(ILUT_OPENGL);

	// Generate the main image name to use.
	ILuint imgId;
	ilGenImages(1, &imgId);

	// Bind this image name.
	ilBindImage(imgId);

	// Loads the image specified by file into the image named by imgId.
	if (!ilLoadImage(input_image_filename.c_str()))
	{
		std::cout << "Fail to load an image file" << std::endl;
		return;
	}

	//
	std::cout << "Width: " << ilGetInteger(IL_IMAGE_WIDTH) << ",  height: " << ilGetInteger(IL_IMAGE_HEIGHT) << ",  depth: " << ilGetInteger(IL_IMAGE_DEPTH) << ",  Bpp: " << ilGetInteger(IL_IMAGE_BITS_PER_PIXEL) << std::endl;

	// Enable this to let us overwrite the destination file if it already exists.
	ilEnable(IL_FILE_OVERWRITE);

	// Save an image.
	if (!ilSaveImage(output_image_filename.c_str()))
	{
		std::cout << "Fail to save an image file" << std::endl;
		return;
	}

	// We're done with the image, so let's delete it.
	ilDeleteImages(1, &imgId);
}

void load_and_save_in_cpp()
{
#if defined(UNICODE) || defined(_UNICODE)
	const std::wstring input_image_filename(L"./data/graphics_2d/devil/640-5.jpg");
	const std::wstring output_image_filename(input_image_filename + L"_cpp.png");
#else
	const std::string input_image_filename("./data/graphics_2d/devil/640-5.jpg");
	const std::string output_image_filename(input_image_filename + "_cpp.png");
#endif

	if (ilGetInteger(IL_VERSION_NUM) < IL_VERSION ||
		iluGetInteger(ILU_VERSION_NUM) < ILU_VERSION ||
		ilutGetInteger(ILUT_VERSION_NUM) < ILUT_VERSION)
	{
		std::cout << "DevIL library is out of date! Please upgrade" << std::endl;
		return;
	}

	ilImage image;

	// Loads an image.
	if (!image.Load(input_image_filename.c_str()))
	{
		std::cout << "Fail to load an image file" << std::endl;
		return;
	}

	//
	std::cout << "Width: " << image.Width() << ",  height: " << image.Height() << ",  depth: " << image.Depth() << ",  Bpp: " << (ILuint)image.Bitpp() << std::endl;

	// Enable this to let us overwrite the destination file if it already exists.
	ilEnable(IL_FILE_OVERWRITE);

	// Save an image.
	if (!image.Save(output_image_filename.c_str()))
	{
		std::cout << "Fail to save an image file" << std::endl;
		return;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_devil {

void basic_operation()
{
	local::load_and_save_in_c();
	local::load_and_save_in_cpp();
}

}  // namespace my_devil
