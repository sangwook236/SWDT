/*
 * =====================================================================================
 *
 *       Filename:  bskde.h
 *
 *    Description:  Background subtract using Kernel Density Estimation 
 *
 *        Version:  1.0
 *        Created:  07/15/2010 09:42:32 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Jing (gk), race604@gmail.com
 *        Company:  BUPT, Beijing
 *
 * =====================================================================================
 */

#include	<cv.h>
#include	<cxcore.h>
#include	<cvaux.h>

#define e 2.7183
#define pi 3.14
#define numFrame 64

class BS_KDE
{
private:
	double segama;
	int numInitial;
	int numMRF;
	int num_write;
	double threshold;
	double alpha;
	int frameIndex; //Frame counter
	double hash[256];

	IplImage* frame; //original images of video 
	IplImage* frame_gray; //original images of video of gray level 
	IplImage* frame_temp; //next frame to be computed 

	IplImage* frame_result; // the result after background subtraction 
	IplImage* chain[numFrame]; //save the images to compute the kernel 
	IplImage* diff[numFrame]; //save the absolute difference 
	IplImage* phi[numFrame]; //save the MRF temp results 
	IplImage* dividend;  
	//IplImage* abc; 
	IplImage* frame_left_shift; //save the image with 1 pixel shifted to the left 
	IplImage* frame_right_shift; //save the image with 1 pixel shifted to the right 
	IplImage* frame_up_shift; //save the image with 1 pixel shifted to the up 
	IplImage* frame_down_shift; //save the image with 1 pixel shifted to the down 
	IplImage* frame_MRF; //save the MRF 
	IplImage* frame_diff; //save the difference of the image 

	CvSize size;
	CvSize size_shift;

	double gaussian_value(int value);

public:
	BS_KDE();
	int ProcessFrame(IplImage* newImg, IplImage* pFgImg=NULL, IplImage* mask=NULL);

};
