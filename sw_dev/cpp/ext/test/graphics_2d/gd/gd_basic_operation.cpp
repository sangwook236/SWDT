#include <gd.h>
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
	// Allocate the image: 64 pixels across by 64 pixels tall.
	gdImagePtr im = gdImageCreate(64, 64);

	// Allocate the color black (red, green and blue all minimum). Since this is the first color in a new image, it will be the background color.
	const int black = gdImageColorAllocate(im, 0, 0, 0);

	// Allocate the color white (red, green and blue all maximum).
	const int white = gdImageColorAllocate(im, 255, 255, 255);

	// Draw a line from the upper left to the lower right, using white color index.
	gdImageLine(im, 0, 0, 63, 63, white);

	// Open a file for writing. "wb" means "write binary", important under MSDOS, harmless under Unix.
	FILE *pngout = std::fopen("./data/graphics_2d/gd/test.png", "wb");
	if (NULL == pngout)
	{
		if (EACCES == errno)
			std::cout << "Error access." << std::endl;
		else if (EISDIR == errno)
			std::cout << "Error directory." << std::endl;
		else if (ENOENT == errno)
			std::cout << "Error file." << std::endl;

		return;
	}

	// Do the same for a JPEG-format file.
	FILE *jpegout = std::fopen("./data/graphics_2d/gd/test.jpg", "wb");
	if (NULL == jpegout)
	{
		if (EACCES == errno)
			std::cout << "Error access." << std::endl;
		else if (EISDIR == errno)
			std::cout << "Error directory." << std::endl;
		else if (ENOENT == errno)
			std::cout << "Error file." << std::endl;

		return;
	}

	// Output the image to the disk file in PNG format.
	gdImagePng(im, pngout);

	// Output the same image in JPEG format, using the default JPEG quality setting.
	gdImageJpeg(im, jpegout, -1);

	// Close the files.
	std::fclose(pngout);
	std::fclose(jpegout);

	// Destroy the image in memory.
	gdImageDestroy(im);
}

}  // namespace local
}  // unnamed namespace

#if defined(__cplusplus)
}
#endif

namespace my_gd {

void basic_operation()
{
	local::load_and_save();
}

}  // namespace my_gd
