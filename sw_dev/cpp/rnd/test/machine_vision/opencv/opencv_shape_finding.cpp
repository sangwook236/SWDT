//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <list>


namespace {
namespace local {

void contour(IplImage *srcImg, IplImage *grayImg)
{
	const int levels = 5;
	CvSeq *contours = NULL;
    CvMemStorage *storage = cvCreateMemStorage(0);

    cvFindContours(grayImg, storage, &contours, sizeof(CvContour), CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

    // comment this out if you do not want approximation
    contours = cvApproxPoly(contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, 3, 1);

#if 0
	const int _levels = levels - 3;
	CvSeq *_contours = contours;
    if (_levels <= 0)  // get to the nearest face to make it look more funny
        _contours = _contours->h_next->h_next->h_next;

	cvDrawContours(srcImg, _contours, CV_RGB(255, 0, 0), CV_RGB(0, 255, 0), _levels, 3, CV_AA, cvPoint(0, 0));
#else
	cvDrawContours(srcImg, contours, CV_RGB(255, 0, 0), CV_RGB(0, 255, 0), levels, 3, CV_AA, cvPoint(0, 0));
#endif

    cvReleaseMemStorage(&storage);
}

void ellipse(IplImage *srcImg, IplImage *grayImg)
{
	const int threshold = 70;

	// create dynamic structure and sequence.
	CvMemStorage *storage = cvCreateMemStorage(0);
	CvSeq *contour = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage);

	IplImage *img = cvCloneImage(grayImg);

	// threshold the source image. this needful for cvFindContours().
	cvThreshold(grayImg, img, threshold, 255, CV_THRESH_BINARY);

	// find all contours.
	cvFindContours(img, storage, &contour, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_NONE, cvPoint(0,0));

	// this cycle draw all contours and approximate it by ellipses.
	for ( ; contour; contour = contour->h_next)
	{
		const int count = contour->total;  // this is number point in contour

		// number point must be more than or equal to 6 (for cvFitEllipse_32f).
		if (count < 6) continue;

		CvMat *points_f = cvCreateMat(1, count, CV_32FC2);
		CvMat points_i = cvMat(1, count, CV_32SC2, points_f->data.ptr);
		cvCvtSeqToArray(contour, points_f->data.ptr, CV_WHOLE_SEQ);
		cvConvert(&points_i, points_f);

		// fits ellipse to current contour.
		const CvBox2D box = cvFitEllipse2(points_f);

		// draw current contour.
		cvDrawContours(srcImg, contour, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), 0, 2, 8, cvPoint(0, 0));

		// convert ellipse data from float to integer representation.
		const CvPoint center = cvPointFrom32f(box.center);
		CvSize size;
		size.width = cvRound(box.size.width * 0.5);
		size.height = cvRound(box.size.height * 0.5);

		// draw ellipse.
		cvEllipse(srcImg, center, size, -box.angle, 0, 360, CV_RGB(255, 0, 0), 2, CV_AA, 0);

		cvReleaseMat(&points_f);
	}

	cvReleaseImage(&img);
}

// helper function:
// finds a cosine of angle between vectors from pt0->pt1 and from pt0->pt2
double calc_angle(const CvPoint *pt1, const CvPoint *pt2, const CvPoint *pt0)
{
    const double dx1 = pt1->x - pt0->x;
    const double dy1 = pt1->y - pt0->y;
    const double dx2 = pt2->x - pt0->x;
    const double dy2 = pt2->y - pt0->y;
	return (dx1*dx2 + dy1*dy2) / std::sqrt((dx1*dx1 + dy1*dy1) * (dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
CvSeq * find_rectangle(IplImage *img, CvMemStorage *storage)
{
	const double threshold1 = 0.0;
	const double threshold2 = 50.0;
	const int aperture_size = 5;
	const int N = 11;

	const CvSize sz = cvSize(img->width & -2, img->height & -2);
	IplImage *timg = cvCloneImage(img);  // make a copy of input image
	IplImage *gray = cvCreateImage(sz, 8, 1);
	IplImage *tgray = cvCreateImage(sz, 8, 1);

	// create empty sequence that will contain points - 4 points per square (the square's vertices)
	CvSeq *squares = cvCreateSeq(0, sizeof(CvSeq), sizeof(CvPoint), storage);

	// select the maximum ROI in the image with the width and height divisible by 2
	cvSetImageROI(timg, cvRect(0, 0, sz.width, sz.height));

	// down-scale and up-scale the image to filter out the noise
	IplImage *pyr = cvCreateImage(cvSize(sz.width/2, sz.height/2), 8, 3);
	cvPyrDown(timg, pyr, CV_GAUSSIAN_5x5);
	cvPyrUp(pyr, timg, CV_GAUSSIAN_5x5);
	cvReleaseImage(&pyr);

	// find squares in every color plane of the image
	double s, t;
	for (int c = 0; c < 3; ++c)
	{
		// extract the c-th color plane
		cvSetImageCOI(timg, c+1);
		cvCopy(timg, tgray, NULL);

		// try several threshold levels
		for (int l = 0; l < N; ++l)
		{
			// hack: use Canny instead of zero threshold level.
			// Canny helps to catch squares with gradient shading
			if (l == 0)
			{
				// apply Canny. take the upper threshold from slider and set the lower to 0 (which forces edges merging)
				cvCanny(tgray, gray, threshold1, threshold2, aperture_size);
				// dilate canny output to remove potential holes between edge segments
				cvDilate(gray, gray, NULL, 1);
			}
			else
			{
				// apply threshold if l != 0: tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
				cvThreshold(tgray, gray, (l+1)*255/N, 255, CV_THRESH_BINARY);
			}

			// find contours and store them all as a list
			CvSeq *contours = NULL;
			cvFindContours(gray, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

			// test each contour
			while (contours)
			{
				// approximate contour with accuracy proportional to the contour perimeter
				CvSeq *result = cvApproxPoly(contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);
				// square contours should have 4 vertices after approximation relatively large area (to filter out noisy contours) and be convex.
				// Note: absolute value of an area is used because area may be positive or negative - in accordance with the contour orientation
				if (result->total == 4 && cvContourArea(result, CV_WHOLE_SEQ) > 1000 && cvCheckContourConvexity(result))
				{
					s = 0;

					for (int i = 0; i < 5; ++i)
					{
						// find minimum angle between joint edges (maximum of cosine)
						if (i >= 2)
						{
							t = std::fabs(calc_angle((CvPoint *)cvGetSeqElem(result, i), (CvPoint *)cvGetSeqElem(result, i-2), (CvPoint *)cvGetSeqElem(result, i-1)));
							s = s > t ? s : t;
						}
					}

					// if cosines of all angles are small (all angles are ~90 degree) then write quandrange vertices to resultant sequence
					if (s < 0.3)
						for (int i = 0; i < 4; ++i)
							cvSeqPush(squares, (CvPoint *)cvGetSeqElem(result, i));
				}

				// take the next contour
				contours = contours->h_next;
			}
		}
	}

	cvResetImageROI(timg);

	// release all the temporary images
	cvReleaseImage(&gray);
	cvReleaseImage(&tgray);
	cvReleaseImage(&timg);

	return squares;
}


// the function draws all the squares in the image
void draw_rectangle(IplImage* img, const CvSeq *squares)
{
	CvSeqReader reader;

	// initialize reader of the sequence
	cvStartReadSeq(squares, &reader, 0);

	// read 4 sequence elements at a time (all vertices of a square)
	for (int i = 0; i < squares->total; i += 4)
	{
		CvPoint pt[4], *rect = pt;
		const int count = 4;

		// read 4 vertices
		CV_READ_SEQ_ELEM(pt[0], reader);
		CV_READ_SEQ_ELEM(pt[1], reader);
		CV_READ_SEQ_ELEM(pt[2], reader);
		CV_READ_SEQ_ELEM(pt[3], reader);

		// draw the square as a closed polyline
		cvPolyLine(img, &rect, &count, 1, 1, CV_RGB(255, 0, 0), 3, CV_AA, 0);
	}
}

void rectangle(IplImage *srcImg)
{
	CvMemStorage *storage = cvCreateMemStorage(0);

	const CvSeq *rectangles = find_rectangle(srcImg, storage);
	draw_rectangle(srcImg, rectangles);

	cvClearMemStorage(storage);
	cvReleaseMemStorage(&storage);
}

}  // namespace local
}  // unnamed namespace

namespace opencv {

void shape_finding()
{
	std::list<std::string> filenames;
	filenames.push_back("machine_vision_data\\opencv\\pic1.png");
	filenames.push_back("machine_vision_data\\opencv\\pic2.png");
	filenames.push_back("machine_vision_data\\opencv\\pic3.png");
	filenames.push_back("machine_vision_data\\opencv\\pic4.png");
	filenames.push_back("machine_vision_data\\opencv\\pic5.png");
	filenames.push_back("machine_vision_data\\opencv\\pic6.png");
	filenames.push_back("machine_vision_data\\opencv\\stuff.jpg");
	filenames.push_back("machine_vision_data\\opencv\\synthetic_face.png");
	filenames.push_back("machine_vision_data\\opencv\\puzzle.png");
	filenames.push_back("machine_vision_data\\opencv\\fruits.jpg");
	filenames.push_back("machine_vision_data\\opencv\\lena_rgb.bmp");
	filenames.push_back("machine_vision_data\\opencv\\hand_01.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_05.jpg");
	filenames.push_back("machine_vision_data\\opencv\\hand_24.jpg");

	const char *windowName = "shape finding";
	cvNamedWindow(windowName, CV_WINDOW_AUTOSIZE);

	//
	for (std::list<std::string>::iterator it = filenames.begin(); it != filenames.end(); ++it)
    {

		IplImage *srcImage = cvLoadImage(it->c_str());
		if (NULL == srcImage)
		{
			std::cout << "fail to load image file: " << *it << std::endl;
			continue;
		}

		IplImage *grayImage = NULL;
		if (1 == srcImage->nChannels)
			cvCopy(srcImage, grayImage, NULL);
		else
		{
			grayImage = cvCreateImage(cvGetSize(srcImage), srcImage->depth, 1);
#if defined(__GNUC__)
			if (strcasecmp(srcImage->channelSeq, "RGB") == 0)
#elif defined(_MSC_VER)
			if (_stricmp(srcImage->channelSeq, "RGB") == 0)
#endif
				cvCvtColor(srcImage, grayImage, CV_RGB2GRAY);
#if defined(__GNUC__)
			else if (strcasecmp(srcImage->channelSeq, "BGR") == 0)
#elif defined(_MSC_VER)
			else if (_stricmp(srcImage->channelSeq, "BGR") == 0)
#endif
				cvCvtColor(srcImage, grayImage, CV_BGR2GRAY);
			else
				assert(false);
			grayImage->origin = srcImage->origin;
		}

		//
		local::contour(srcImage, grayImage);
		//local::ellipse(srcImage, grayImage);
		//local::rectangle(srcImage);
		//local::snake(srcImage, grayImage);

		//
		cvShowImage(windowName, srcImage);

		const unsigned char key = cvWaitKey(0);
		if (27 == key)
			break;

		//
		cvReleaseImage(&grayImage);
		cvReleaseImage(&srcImage);
	}

	cvDestroyWindow(windowName);
}

}  // namespace opencv
