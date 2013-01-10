//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <list>
#include <cstdio>


namespace {
namespace local {

void watershed_help()
{
	std::cout << "\nThis program demonstrates the famous watershed segmentation algorithm in OpenCV: watershed()\n"
			"Usage:\n"
			"./watershed [image_name -- default is fruits.jpg]\n" << std::endl;


	std::cout << "Hot keys: \n"
		"\tESC - quit the program\n"
		"\tr - restore the original image\n"
		"\tw or SPACE - run watershed segmentation algorithm\n"
		"\t\t(before running it, *roughly* mark the areas to segment on the image)\n"
		"\t  (before that, roughly outline several markers on the image)\n";
}

cv::Mat watershed_markerMask, watershed_img;
cv::Point watershed_prevPt(-1, -1);

void watershed_onMouse( int event, int x, int y, int flags, void* )
{
    if( x < 0 || x >= watershed_img.cols || y < 0 || y >= watershed_img.rows )
        return;
    if( event == CV_EVENT_LBUTTONUP || !(flags & CV_EVENT_FLAG_LBUTTON) )
        watershed_prevPt = cv::Point(-1,-1);
    else if( event == CV_EVENT_LBUTTONDOWN )
        watershed_prevPt = cv::Point(x,y);
    else if( event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON) )
    {
        cv::Point pt(x, y);
        if( watershed_prevPt.x < 0 )
            watershed_prevPt = pt;
        line( watershed_markerMask, watershed_prevPt, pt, cv::Scalar::all(255), 5, 8, 0 );
        line( watershed_img, watershed_prevPt, pt, cv::Scalar::all(255), 5, 8, 0 );
        watershed_prevPt = pt;
        cv::imshow("image", watershed_img);
    }
}

void watershed_algorithm(const cv::Mat &img0)
{
    watershed_help();
	cv::namedWindow( "image", 1 );

	cv::Mat imgGray;

    img0.copyTo(watershed_img);

#if 0
	// use gradient
	cv::Mat gray;
    cv::cvtColor(img0, gray, CV_BGR2GRAY);

	const int ksize = 5;
	cv::Mat xgradient, ygradient;
	cv::Sobel(gray, xgradient, CV_32FC1, 1, 0, ksize, 1.0, 0.0);
	cv::Sobel(gray, ygradient, CV_32FC1, 0, 1, ksize, 1.0, 0.0);

	cv::Mat gradient;
	cv::magnitude(xgradient, ygradient, gradient);

	double minVal = 0.0, maxVal = 0.0;
	cv::minMaxLoc(gradient, &minVal, &maxVal);

	cv::Mat gradient8u;
	gradient.convertTo(gradient8u, CV_8UC1, 255.0 / (maxVal - minVal), 255.0 * minVal / (maxVal - minVal));
    cv::cvtColor(gradient8u, watershed_img, CV_GRAY2BGR);
#endif

    cv::cvtColor(watershed_img, watershed_markerMask, CV_BGR2GRAY);
    cv::cvtColor(watershed_markerMask, imgGray, CV_GRAY2BGR);
    watershed_markerMask = cv::Scalar::all(0);
    cv::imshow( "image", watershed_img );
	cv::setMouseCallback( "image", watershed_onMouse, 0 );

    for(;;)
    {
        int c = cv::waitKey(0);

        if( (char)c == 27 )
            break;

        if( (char)c == 'r' )
        {
            watershed_markerMask = cv::Scalar::all(0);
            img0.copyTo(watershed_img);
            cv::imshow( "image", watershed_img );
        }

        if( (char)c == 'w' || (char)c == ' ' )
        {
            int i, j, compCount = 0;
            std::vector<std::vector<cv::Point> > contours;
            std::vector<cv::Vec4i> hierarchy;

            cv::findContours(watershed_markerMask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

            if( contours.empty() )
                continue;
            cv::Mat markers(watershed_markerMask.size(), CV_32S);
            markers = cv::Scalar::all(0);
            int idx = 0;
            for( ; idx >= 0; idx = hierarchy[idx][0], compCount++ )
                cv::drawContours(markers, contours, idx, cv::Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX);

            if( compCount == 0 )
                continue;

            std::vector<cv::Vec3b> colorTab;
            for( i = 0; i < compCount; i++ )
            {
                int b = cv::theRNG().uniform(0, 255);
                int g = cv::theRNG().uniform(0, 255);
                int r = cv::theRNG().uniform(0, 255);

                colorTab.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
            }

			double t = (double)cv::getTickCount();
            watershed( img0, markers );
            t = (double)cv::getTickCount() - t;
            printf( "execution time = %gms\n", t*1000./cv::getTickFrequency() );

            cv::Mat wshed(markers.size(), CV_8UC3);

            // paint the watershed image
            for( i = 0; i < markers.rows; i++ )
                for( j = 0; j < markers.cols; j++ )
                {
                    int idx = markers.at<int>(i,j);
                    if( idx == -1 )
                        wshed.at<cv::Vec3b>(i,j) = cv::Vec3b(255,255,255);
                    else if( idx <= 0 || idx > compCount )
                        wshed.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
                    else
                        wshed.at<cv::Vec3b>(i,j) = colorTab[idx - 1];
                }

            wshed = wshed*0.5 + imgGray*0.5;
            cv::imshow( "watershed transform", wshed );
        }
    }
}

void grabcut_help()
{
    std::cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
    		"and then grabcut will attempt to segment it out.\n"
    		"Call:\n"
    		"./grabcut <image_name>\n"
    	"\nSelect a rectangular area around the object you want to segment\n" <<
        "\nHot keys: \n"
        "\tESC - quit the program\n"
        "\tr - restore the original image\n"
        "\tn - next iteration\n"
        "\n"
        "\tleft mouse button - set rectangle\n"
        "\n"
        "\tCTRL+left mouse button - set GC_BGD pixels\n"
        "\tSHIFT+left mouse button - set CG_FGD pixels\n"
        "\n"
        "\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
        "\tSHIFT+right mouse button - set CG_PR_FGD pixels\n" << std::endl;
}

const cv::Scalar RED = cv::Scalar(0,0,255);
const cv::Scalar PINK = cv::Scalar(230,130,255);
const cv::Scalar BLUE = cv::Scalar(255,0,0);
const cv::Scalar LIGHTBLUE = cv::Scalar(255,255,160);
const cv::Scalar GREEN = cv::Scalar(0,255,0);

const int BGD_KEY = CV_EVENT_FLAG_CTRLKEY;
const int FGD_KEY = CV_EVENT_FLAG_SHIFTKEY;

void getBinMask( const cv::Mat& comMask, cv::Mat& binMask )
{
    if( comMask.empty() || comMask.type()!=CV_8UC1 )
        CV_Error( CV_StsBadArg, "comMask is empty or has incorrect type (not CV_8UC1)" );
    if( binMask.empty() || binMask.rows!=comMask.rows || binMask.cols!=comMask.cols )
        binMask.create( comMask.size(), CV_8UC1 );
    binMask = comMask & 1;
}

class GCApplication
{
public:
    enum{ NOT_SET = 0, IN_PROCESS = 1, SET = 2 };
    static const int radius = 2;
    static const int thickness = -1;

    void reset();
    void setImageAndWinName( const cv::Mat& _image, const std::string& _winName );
    void showImage() const;
    void mouseClick( int event, int x, int y, int flags, void* param );
    int nextIter();
    int getIterCount() const { return iterCount; }
private:
    void setRectInMask();
    void setLblsInMask( int flags, cv::Point p, bool isPr );

    const std::string* winName;
    const cv::Mat* image;
    cv::Mat mask;
    cv::Mat bgdModel, fgdModel;

    uchar rectState, lblsState, prLblsState;
    bool isInitialized;

    cv::Rect rect;
    std::vector<cv::Point> fgdPxls, bgdPxls, prFgdPxls, prBgdPxls;
    int iterCount;
};

void GCApplication::reset()
{
    if( !mask.empty() )
		mask.setTo(cv::Scalar::all(cv::GC_BGD));
    bgdPxls.clear(); fgdPxls.clear();
    prBgdPxls.clear();  prFgdPxls.clear();

    isInitialized = false;
    rectState = NOT_SET;
    lblsState = NOT_SET;
    prLblsState = NOT_SET;
    iterCount = 0;
}

void GCApplication::setImageAndWinName( const cv::Mat& _image, const std::string& _winName  )
{
    if( _image.empty() || _winName.empty() )
        return;
    image = &_image;
    winName = &_winName;
    mask.create( image->size(), CV_8UC1);
    reset();
}

void GCApplication::showImage() const
{
    if( image->empty() || winName->empty() )
        return;

    cv::Mat res;
    cv::Mat binMask;
    if( !isInitialized )
        image->copyTo( res );
    else
    {
        getBinMask( mask, binMask );
        image->copyTo( res, binMask );
    }

    std::vector<cv::Point>::const_iterator it;
    for( it = bgdPxls.begin(); it != bgdPxls.end(); ++it )
        circle( res, *it, radius, BLUE, thickness );
    for( it = fgdPxls.begin(); it != fgdPxls.end(); ++it )
        circle( res, *it, radius, RED, thickness );
    for( it = prBgdPxls.begin(); it != prBgdPxls.end(); ++it )
        circle( res, *it, radius, LIGHTBLUE, thickness );
    for( it = prFgdPxls.begin(); it != prFgdPxls.end(); ++it )
        circle( res, *it, radius, PINK, thickness );

    if( rectState == IN_PROCESS || rectState == SET )
        rectangle( res, cv::Point( rect.x, rect.y ), cv::Point(rect.x + rect.width, rect.y + rect.height ), GREEN, 2);

    cv::imshow( *winName, res );
}

void GCApplication::setRectInMask()
{
    assert( !mask.empty() );
	mask.setTo( cv::GC_BGD );
	rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width = std::min(rect.width, image->cols-rect.x);
    rect.height = std::min(rect.height, image->rows-rect.y);
	(mask(rect)).setTo( cv::Scalar(cv::GC_PR_FGD) );
}

void GCApplication::setLblsInMask( int flags, cv::Point p, bool isPr )
{
    std::vector<cv::Point> *bpxls, *fpxls;
    uchar bvalue, fvalue;
    if( !isPr )
    {
        bpxls = &bgdPxls;
        fpxls = &fgdPxls;
        bvalue = cv::GC_BGD;
        fvalue = cv::GC_FGD;
    }
    else
    {
        bpxls = &prBgdPxls;
        fpxls = &prFgdPxls;
        bvalue = cv::GC_PR_BGD;
        fvalue = cv::GC_PR_FGD;
    }
    if( flags & BGD_KEY )
    {
        bpxls->push_back(p);
        circle( mask, p, radius, bvalue, thickness );
    }
    if( flags & FGD_KEY )
    {
        fpxls->push_back(p);
        circle( mask, p, radius, fvalue, thickness );
    }
}

void GCApplication::mouseClick( int event, int x, int y, int flags, void* )
{
    // TODO add bad args check
    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels
        {
            bool isb = (flags & BGD_KEY) != 0,
                 isf = (flags & FGD_KEY) != 0;
            if( rectState == NOT_SET && !isb && !isf )
            {
                rectState = IN_PROCESS;
                rect = cv::Rect( x, y, 1, 1 );
            }
            if ( (isb || isf) && rectState == SET )
                lblsState = IN_PROCESS;
        }
        break;
    case CV_EVENT_RBUTTONDOWN: // set GC_PR_BGD(GC_PR_FGD) labels
        {
            bool isb = (flags & BGD_KEY) != 0,
                 isf = (flags & FGD_KEY) != 0;
            if ( (isb || isf) && rectState == SET )
                prLblsState = IN_PROCESS;
        }
        break;
    case CV_EVENT_LBUTTONUP:
        if( rectState == IN_PROCESS )
        {
            rect = cv::Rect( cv::Point(rect.x, rect.y), cv::Point(x,y) );
            rectState = SET;
            setRectInMask();
            assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
            showImage();
        }
        if( lblsState == IN_PROCESS )
        {
            setLblsInMask(flags, cv::Point(x,y), false);
            lblsState = SET;
            showImage();
        }
        break;
    case CV_EVENT_RBUTTONUP:
        if( prLblsState == IN_PROCESS )
        {
            setLblsInMask(flags, cv::Point(x,y), true);
            prLblsState = SET;
            showImage();
        }
        break;
    case CV_EVENT_MOUSEMOVE:
        if( rectState == IN_PROCESS )
        {
            rect = cv::Rect( cv::Point(rect.x, rect.y), cv::Point(x,y) );
            assert( bgdPxls.empty() && fgdPxls.empty() && prBgdPxls.empty() && prFgdPxls.empty() );
            showImage();
        }
        else if( lblsState == IN_PROCESS )
        {
            setLblsInMask(flags, cv::Point(x,y), false);
            showImage();
        }
        else if( prLblsState == IN_PROCESS )
        {
            setLblsInMask(flags, cv::Point(x,y), true);
            showImage();
        }
        break;
    }
}

int GCApplication::nextIter()
{
    if( isInitialized )
        grabCut( *image, mask, rect, bgdModel, fgdModel, 1 );
    else
    {
        if( rectState != SET )
            return iterCount;

        if( lblsState == SET || prLblsState == SET )
            grabCut( *image, mask, rect, bgdModel, fgdModel, 1, cv::GC_INIT_WITH_MASK );
        else
            grabCut( *image, mask, rect, bgdModel, fgdModel, 1, cv::GC_INIT_WITH_RECT );

        isInitialized = true;
    }
    iterCount++;

    bgdPxls.clear(); fgdPxls.clear();
    prBgdPxls.clear(); prFgdPxls.clear();

    return iterCount;
}

GCApplication gcapp;

void grabcut_on_mouse( int event, int x, int y, int flags, void* param )
{
    gcapp.mouseClick( event, x, y, flags, param );
}

void grabcut_algorithm(const cv::Mat &image)
{
    grabcut_help();

    const std::string winName = "image";
    cvNamedWindow( winName.c_str(), CV_WINDOW_AUTOSIZE );
    cvSetMouseCallback( winName.c_str(), grabcut_on_mouse, 0 );

    gcapp.setImageAndWinName( image, winName );
    gcapp.showImage();

    for(;;)
    {
        int c = cvWaitKey(0);
        switch( (char) c )
        {
        case '\x1b':
            std::cout << "Exiting ..." << std::endl;
            goto exit_main;
        case 'r':
            std::cout << std::endl;
            gcapp.reset();
            gcapp.showImage();
            break;
        case 'n':
            int iterCount = gcapp.getIterCount();
            std::cout << "<" << iterCount << "... ";
            int newIterCount = gcapp.nextIter();
            if( newIterCount > iterCount )
            {
                gcapp.showImage();
                std::cout << iterCount << ">" << std::endl;
            }
            else
                std::cout << "rect must be determined>" << std::endl;
            break;
        }
    }

exit_main:
    cvDestroyWindow( winName.c_str() );
    return;
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void segmentation()
{
	//const std::string filename("machine_vision_data\\opencv\\pic1.png");
	//const std::string filename("machine_vision_data\\opencv\\pic2.png");
	//const std::string filename("machine_vision_data\\opencv\\pic3.png");
	//const std::string filename("machine_vision_data\\opencv\\pic4.png");
	//const std::string filename("machine_vision_data\\opencv\\pic5.png");
	//const std::string filename("machine_vision_data\\opencv\\pic6.png");
	//const std::string filename("machine_vision_data\\opencv\\stuff.jpg");
	//const std::string filename("machine_vision_data\\opencv\\synthetic_face.png");
	//const std::string filename("machine_vision_data\\opencv\\puzzle.png");
	//const std::string filename("machine_vision_data\\opencv\\fruits.jpg");
	//const std::string filename("machine_vision_data\\opencv\\lena_rgb.bmp");
	//const std::string filename("machine_vision_data\\opencv\\hand_01.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_05.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_24.jpg");

	//const std::string filename("machine_vision_data\\opencv\\hand_01.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_02.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_03.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_04.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_05.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_06.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_07.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_08.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_09.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_10.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_11.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_12.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_13.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_14.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_15.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_16.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_17.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_18.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_19.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_20.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_21.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_22.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_23.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_24.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_25.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_26.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_27.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_28.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_29.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_30.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_31.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_32.jpg");
	const std::string filename("machine_vision_data\\opencv\\hand_33.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_34.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_35.jpg");
	//const std::string filename("machine_vision_data\\opencv\\hand_36.jpg");

	const std::string windowName("segmentation");
	//cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	//
	cv::Mat img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	if (img.empty())
	{
		std::cout << "fail to load image file: " << filename << std::endl;
		return;
	}

	local::watershed_algorithm(img);
	//local::grabcut_algorithm(img);

	//
	//cv::imshow(windowName, img);

	cv::waitKey(0);

	//cv::destroyWindow(windowName);
}

}  // namespace my_opencv
