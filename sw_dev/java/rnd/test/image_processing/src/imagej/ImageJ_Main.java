package imagej;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.OvalRoi;
import ij.gui.Roi;
import ij.io.FileSaver;
import ij.io.Opener;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;

public class ImageJ_Main {

	public static void run(String[] args)
	{
		// Basic operation.
		{
			// REF [site] >> http://albert.rierol.net/imagej_programming_tutorials.html#ImageJ programming basics
			
			//createImage();
			openImage();
			//saveImage();
			//editImage();
		}
	}

	static void createImage()
	{
		int width = 400;
		int height = 400;

		//
		ImageProcessor ip = new ByteProcessor(width, height);
		String title = "My image #1";
		ImagePlus imp = new ImagePlus(title, ip);
		imp.show();

		// A call to flush() will release all memory resources used by the ImagePlus.
		imp.flush();

		//
		new ImagePlus("My image #2", new ByteProcessor(width, height)).show();
		
		// A simple 8-bit grayscale image of 400x400 pixels.
		ImagePlus imp1 = IJ.createImage("My image #3", "8-bit black", width, height, 1);
		imp1.show();

		imp1.flush();
		
		// Without getting back a reference:
		IJ.newImage("My image #4", "8-bit black", width, height, 1);
		
		// A stack of 10 color images of 400x400 pixels.
		ImagePlus imp2 = IJ.createImage("My image #5", "RGB white", width, height, 10);
		imp2.show();

		imp2.flush();

		// Without getting back a reference:
		IJ.newImage("My image #6", "RGB white", width, height, 10);

		// A call to flush() will release all memory resources used by the ImagePlus.
		imp.flush();
	}
	
	static void openImage()
	{
		ImagePlus imp1 = IJ.openImage("data/image_processing/racoon.jpg");
		imp1.show();

		ImagePlus imp2 = IJ.openImage("https://www.gstatic.com/webp/gallery3/1.png");
		imp2.show();

		// Without getting back a pointer, and automatically showing it.
		//IJ.open("data/image_processing/racoon.jpg");  // Exception occurred.
		IJ.open("D:/work/swdt_github/sw_dev/java/rnd/bin/data/image_processing/racoon.jpg");
		// Same but from an URL.
		IJ.open("https://www.gstatic.com/webp/gallery3/1.png");

		Opener opener1 = new Opener();
		ImagePlus imp3 = opener1.openImage("data/image_processing/racoon.jpg");
		imp3.show();

		Opener opener2 = new Opener();
		ImagePlus imp4 = opener2.openImage("https://www.gstatic.com/webp/gallery3/1.png");
		//ImagePlus imp4 = opener2.openURL("https://www.gstatic.com/webp/gallery3/1.png");
		imp4.show();
	}
	
	static void saveImage()
	{
		ImagePlus imp = IJ.openImage("data/image_processing/racoon.jpg");
		IJ.saveAs(imp, "tif", "data/image_processing/image1.tif");

		// Use the file format extension.
		IJ.save(imp, "data/image_processing/image2.tif");
		
		// Use FileSaver.
		new FileSaver(imp).saveAsTiff("data/image_processing/image3.tif");
	}
	
	static void editImage()
	{
		// Run ImageJ commands on an image.
		// High-level.
		{
			ImagePlus imp = IJ.openImage("data/image_processing/racoon.jpg");

			// Make a binary image.
			IJ.run(imp, "Convert to Mask", "");  // "" means no arguments.

			// Resize and open a copy in a new window (the 'create' command keyword).
			IJ.run(imp, "Scale...", "x=0.5 y=0.5 width=344 height=345 interpolate create title=[Scaled version of " + imp.getTitle() + "]");
		}
		
		// Mid-level.
		{
			{
				ImagePlus imp = IJ.openImage("data/image_processing/racoon.jpg");
				ImageProcessor ip = imp.getProcessor();
	
				// Assume 8-bit image.
	
				// Fill a rectangular region with 255 (on grayscale this is white color).
				Roi roi = new Roi(30, 40, 100, 100);  // x, y, width, height of the rectangle.
				ip.setRoi(roi);
				ip.setValue(255);
				ip.fill();
				  
				// Fill an oval region with 255 (white color when grayscale LUT).
				OvalRoi oroi = new OvalRoi(50, 60, 100, 150);  // x, y, width, height of the oval.
				ip.setRoi(oroi);
				ip.setValue(255);
				ip.fill(ip.getMask()); // Notice different fill method.
				                       // Regular fill() would fill the entire bounding box rectangle of the OvalRoi.
				// The method above is valid at least for PolygonRoi and ShapeRoi as well.
	
				// Draw the contour of any region with 255 pixel intensity.
				Roi roi2 = new Roi(30, 40, 100, 100);
				ip.setValue(255);
				ip.draw(roi2);
	
				// Update screen view of the image.
				imp.updateAndDraw();
			}
			
			{
				ImagePlus imp = IJ.openImage("data/image_processing/racoon.jpg");
				ImageProcessor ip = imp.getProcessor();

				ip.flipHorizontal();
				ip.flipVertical();
				ip.rotateLeft();
				ip.rotateRight();

				// Rotate WITHOUT enlarging the canvas to fit.
				double angle = 45;
				ip.setInterpolate(true);  // Bilinear.
				ip.rotate(angle);

				// Rotate ENLARGING the canvas and filling the new areas with background color.
				double angle2 = 45;
				IJ.run(imp, "Arbitrarily...", "angle=" + angle2 + " grid=1 interpolate enlarge");

				// Scale WITHOUT modifying the canvas dimensions.
				ip.setInterpolate(true);  // Bilinear.
				ip.scale(2.0, 2.0);  // In X and Y.

				// Scale ENLARGING or SHRINKING the canvas dimensions.
				double sx = 2.0;
				double sy = 0.75;
				int new_width = (int)(ip.getWidth() * sx);
				int new_height = (int)(ip.getHeight() * sy);
				ip.setInterpolate(true);  // Bilinear.
				ImageProcessor ip2 = ip.resize(new_width, new_height);  // Of the same type as the original.
				imp.setProcessor(imp.getTitle(), ip2);  // UPDATE the original ImagePlus.

				// Update screen view of the image.
				imp.updateAndDraw();
			}
		}
		
		// Low-level.
		{
			ImagePlus imp = IJ.openImage("data/image_processing/racoon.jpg");
			ImageProcessor ip = imp.getProcessor();

			// Edit the pixel array.
			if (imp.getType() == ImagePlus.GRAY8)
			{
			    byte[] pixels = (byte[])ip.getPixels();
			    // Do whatever operations directly on the pixel array.
			}

			// Replace the pixel array: ONLY if same size.
			if (imp.getType() == ImagePlus.GRAY8)
			{
			    int width = ip.getWidth();
			    int height = ip.getHeight();
			    byte[] new_pixels = new byte[width * height];
			    // Set each pixel value to whatever, between -128 and 127.
			    for (int y = 0; y < height; ++y)
			    {
			        for (int x = 0; x < width; ++x)
			        {
			            // Edit pixel at x,y position.
			            //new_pixels[y * width + x] = ...;
			        }
			    }
			    // Update ImageProcessor to new array.
			    ip.setPixels(new_pixels);
			}

			// Replace the pixel array but of different length: for example, to resize 2.5 times in width and height.  
			int new_width = (int)(ip.getWidth() * 2.5);
			int new_height = (int)(ip.getHeight() * 2.5);
			ImageProcessor ip2 = ip.createProcessor(new_width, new_height);  // Of same type.
			imp.setProcessor(imp.getTitle(), ip2);  

			if (imp.getType() == ImagePlus.GRAY8)
			{
			    int width = ip.getWidth();
			    int height = ip.getHeight();
			    byte[] pix = (byte[])imp.getProcessor().getPixels();  // Or ip2.getPixels();
			    // Process pixels.
			    for (int y = 0; y < height; ++y)
			    {
			        for (int x = 0; x < width; ++x)
			        {
			            // Edit pixel at x,y position.
			            //new_pixels[y * width + x] = ...;
			        }
			    }
			}

			// DON'T forget to update the screen image.
			imp.updateAndDraw();
		}
	}
}
