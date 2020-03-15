#include <iostream>
#include <string>
#include <cmath>
#include <wand/MagickWand.h>

// REF [site] >>
//	https://imagemagick.org/MagickWand/
//	https://imagemagick.org/api/MagickWand/index.html
//	https://imagemagick.org/

namespace
{
namespace local
{

void throwWandException(MagickWand *wand)
{
	ExceptionType severity;
	char *description = MagickGetException(wand, &severity);
	const std::string msg(GetMagickModule() + description);
	description = (char *)MagickRelinquishMemory(description);
	throw std::runtime_error(msg);
}

void throwViewException(view)
{
	ExceptionType severity;
	char *description = GetWandViewException(view, &severity);                     \
	const std::string msg(GetMagickModule() + description);
	description = (char *)MagickRelinquishMemory(description);
	throw std::runtime_error(msg);
}

// REF [site] >> https://imagemagick.org/script/magick-wand.php
void create_thumbnail_example()
{
	const std::string image_filepath("./image.png");
	const std::string thumbnail_filepath("./thumbnail.png");

	MagickWandGenesis();
	MagickWand *magick_wand = NewMagickWand();

	// Read an image.
	MagickBooleanType status = MagickReadImage(magick_wand, image_filepath.c_str());
	if (MagickFalse == status)
		throwWandException(magick_wand);

	// Turn the images into a thumbnail sequence.
	MagickResetIterator(magick_wand);
	while (MagickNextImage(magick_wand) != MagickFalse)
		MagickResizeImage(magick_wand, 106, 80, LanczosFilter, 1.0);

	// Write the image then destroy it.
	status = MagickWriteImages(magick_wand, thumbnail_filepath.c_str(), MagickTrue);
	if (MagickFalse == status)
		throwWandException(magick_wand);

	magick_wand = DestroyMagickWand(magick_wand);
	MagickWandTerminus();
}

// REF [site] >> https://imagemagick.org/script/magick-wand.php
void contrast_enhancement_example()
{
#define QuantumScale ((MagickRealType)1.0 / (MagickRealType)QuantumRange)
#define SigmoidalContrast(x) \
	(QuantumRange * (1.0 / (1 + std::exp(10.0 * (0.5 - QuantumScale * x))) - 0.0066928509) * 1.0092503)

	const std::string image_filepath("./image.png");
	const std::string sigmoidal_image_filepath("./sigmoidal.png");

	MagickWandGenesis();
	MagickWand *image_wand = NewMagickWand();

	// Read an image.
	MagickBooleanType status = MagickReadImage(image_wand, image_filepath.c_str());
	if (MagickFalse == status)
		throwWandException(image_wand);
	MagickWand *contrast_wand = CloneMagickWand(image_wand);

	// Sigmoidal non-linearity contrast control.
	PixelIterator *iterator = NewPixelIterator(image_wand);
	PixelIterator *contrast_iterator = NewPixelIterator(contrast_wand);
	if ((iterator == (PixelIterator *)NULL) ||
		(contrast_iterator == (PixelIterator *)NULL))
		throwWandException(image_wand);

	MagickPixelPacket pixel;
	long y = 0;
	register long x;
	for (; y < (long)MagickGetImageHeight(image_wand); y++)
	{
		unsigned long width;
		PixelWand **pixels = PixelGetNextIteratorRow(iterator, &width);
		PixelWand **contrast_pixels = PixelGetNextIteratorRow(contrast_iterator, &width);
		if ((pixels == (PixelWand **)NULL) ||
			(contrast_pixels == (PixelWand **)NULL))
			break;
		for (x = 0; x < (long)width; x++)
		{
			PixelGetMagickColor(pixels[x], &pixel);
			pixel.red = SigmoidalContrast(pixel.red);
			pixel.green = SigmoidalContrast(pixel.green);
			pixel.blue = SigmoidalContrast(pixel.blue);
			pixel.index = SigmoidalContrast(pixel.index);
			PixelSetMagickColor(contrast_pixels[x], &pixel);
		}
		(void)PixelSyncIterator(contrast_iterator);
	}
	if (y < (long)MagickGetImageHeight(image_wand))
		throwWandException(image_wand);

	contrast_iterator = DestroyPixelIterator(contrast_iterator);
	iterator = DestroyPixelIterator(iterator);
	image_wand = DestroyMagickWand(image_wand);

	// Write the image then destroy it.
	status = MagickWriteImages(contrast_wand, sigmoidal_image_filepath.c_str(), MagickTrue);
	if (MagickFalse == status)
		throwWandException(image_wand);

	contrast_wand = DestroyMagickWand(contrast_wand);
	MagickWandTerminus();
}

MagickBooleanType SigmoidalContrast(WandView *pixel_view, const ssize_t y, const int id, void *context)
{
#define QuantumScale ((MagickRealType)1.0 / (MagickRealType)QuantumRange)
#define SigmoidalContrast(x) \
	(QuantumRange * (1.0 / (1 + exp(10.0 * (0.5 - QuantumScale * x))) - 0.0066928509) * 1.0092503)

	RectangleInfo extent = GetWandViewExtent(contrast_view);
	PixelWand **pixels = GetWandViewPixels(contrast_view);
	MagickPixelPacket pixel;
	register long x;
	for (x = 0; x < (long)(extent.width - extent.height); x++)
	{
		PixelGetMagickColor(pixels[x], &pixel);
		pixel.red = SigmoidalContrast(pixel.red);
		pixel.green = SigmoidalContrast(pixel.green);
		pixel.blue = SigmoidalContrast(pixel.blue);
		pixel.index = SigmoidalContrast(pixel.index);
		PixelSetMagickColor(contrast_pixels[x], &pixel);
	}
	return (MagickTrue);
}

void contrast_enhancement_example2()
{
	const std::string image_filepath("./image.png");
	const std::string sigmoidal_image_filepath("./sigmoidal.png");

	MagickWandGenesis();
	MagickWand *contrast_wand = NewMagickWand();

	// Read an image.
	MagickBooleanType status = MagickReadImage(contrast_wand, image_filepath.c_str());
	if (MagickFalse == status)
		throwWandException(contrast_wand);

	// Sigmoidal non-linearity contrast control.
	WandView *contrast_view = NewWandView(contrast_wand);
	if (contrast_view == (WandView *)NULL)
		throwWandException(contrast_wand);
	status = UpdateWandViewIterator(contrast_view, SigmoidalContrast, (void *)NULL);
	if (MagickFalse == status)
		throwWandException(contrast_wand);

	contrast_view = DestroyWandView(contrast_view);

	// Write the image then destroy it.
	status = MagickWriteImages(contrast_wand, sigmoidal_image_filepath.c_str(), MagickTrue);
	if (MagickFalse == status)
		throwWandException(contrast_wand);

	contrast_wand = DestroyMagickWand(contrast_wand);
	MagickWandTerminus();
}

} // namespace local
} // unnamed namespace

namespace my_magickwand
{

} // namespace my_magickwand

int magickwand_main(int argc, char *argv[])
{
	// TODO [check] >> Not yet tested.
	local::create_thumbnail_example();
	local::contrast_enhancement_example();
	local::contrast_enhancement_example2();

	return 0;
}
