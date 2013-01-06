#if defined(__linux) || defined(__linux__) || defined(linux) || defined(__unix) || defined(__unix__) || defined(unix)
#include <gd.h>
#else
#include <gd/gd.h>
#endif
#include <iostream>
#include <cstdio>
#include <cerrno>


#if defined(__cplusplus)
extern "C" {
#endif

namespace {
namespace local {

void load_and_save()
{
	// allocate the image: 64 pixels across by 64 pixels tall
	gdImagePtr im = gdImageCreate(64, 64);

	// allocate the color black (red, green and blue all minimum). Since this is the first color in a new image, it will be the background color
	const int black = gdImageColorAllocate(im, 0, 0, 0);

	// allocate the color white (red, green and blue all maximum)
	const int white = gdImageColorAllocate(im, 255, 255, 255);

	// draw a line from the upper left to the lower right, using white color index
	gdImageLine(im, 0, 0, 63, 63, white);

	// open a file for writing. "wb" means "write binary", important under MSDOS, harmless under Unix
	std::FILE *pngout = std::fopen(".\\graphics_2d_data\\gd\\test.png", "wb");
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

	// do the same for a JPEG-format file
	std::FILE *jpegout = std::fopen(".\\graphics_2d_data\\gd\\test.jpg", "wb");
	if (NULL == jpegout)

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

	// output the same image in JPEG format, using the default JPEG quality setting
	gdImageJpeg(im, jpegout, -1);

	// close the files
	std::fclose(pngout);
	std::fclose(jpegout);

	// destroy the image in memory
	gdImageDestroy(im);
}

}  // namespace local
}  // unnamed namespace

#if defined(__cplusplus)
}
#endif

namespace gd {

void basic_operation()
{
	local::load_and_save();
}

}  // namespace gd
