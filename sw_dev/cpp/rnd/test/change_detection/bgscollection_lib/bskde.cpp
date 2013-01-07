/*
 * =====================================================================================
 *
 *       Filename:  bskde.cpp
 *
 *    Description:  Background subtraction with KDE method
 *
 *        Version:  1.0
 *        Created:  07/15/2010 10:36:44 PM
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
#include	"bskde.h"

BS_KDE::BS_KDE()
{
	segama = 30.0;
	numMRF = 5;
	num_write = 500;
	threshold = /* 0.58 */0.4;
	alpha = 0.05;
	frameIndex = 0;

	frame = NULL; //original images of video 
	frame_gray = NULL; //original images of video of gray level 
	frame_temp = NULL; //next frame to be computed 

	frame_result = NULL; // the result after background subtraction 
	frame_left_shift = NULL; //save the image with 1 pixel shifted to the left 
	frame_right_shift = NULL; //save the image with 1 pixel shifted to the right 
	frame_up_shift = NULL; //save the image with 1 pixel shifted to the up 
	frame_down_shift = NULL; //save the image with 1 pixel shifted to the down 
	frame_MRF = NULL; //save the MRF 
	frame_diff = NULL; //save the difference of the image 

	for(int i = 0; i < 256; ++i) 
		hash[i] = gaussian_value(i); 
}

double BS_KDE::gaussian_value(int value) 
{ 
	double temp;
	double index = -pow((double)value, 2)/(2 * pow(segama, 2)); 
	temp = pow(e, index)/numFrame; 
	return temp; 
}

int BS_KDE::ProcessFrame(IplImage* newImg, IplImage* pFgImg, IplImage* mask)
{

	frameIndex++;
	if (frameIndex == 1) // first frame for initialize
	{
		size.width = newImg->width;  size.height = newImg->height;
		size_shift.width = newImg->width;  size_shift.height = newImg->height;
		dividend = cvCreateImage(size, IPL_DEPTH_8U, 1); 
		frame_diff = cvCreateImage(size, IPL_DEPTH_8U, 1); 

		frame_MRF = cvCreateImage(size_shift, IPL_DEPTH_8U, 1); 

		CvScalar value = {numFrame, 0, 0, 0}; 

		cvSet(dividend, value); 

		for(int i = 0; i < numFrame; ++i) 
		{
			phi[i] = cvCreateImage(size, IPL_DEPTH_8U, 1); 
			diff[i] = cvCreateImage(size, IPL_DEPTH_8U, 1); 
		}

		return 0;
	}

	if (frameIndex < numFrame+2)
	{
		//chain[frameIndex-2] = cvCreateImage(size, IPL_DEPTH_8U, 1);
		frame_gray = cvCreateImage(size, IPL_DEPTH_8U, 1);
		cvCvtColor(newImg, frame_gray, CV_RGB2GRAY);
		chain[frameIndex-2] = frame_gray;

		return 0;
	}// initialize 

	time_t TimeStart, TimeEnd, TimeUsed; 

	frame_gray = cvCreateImage(size, IPL_DEPTH_8U, 1); 
		
	TimeStart = clock(); 
	cvCvtColor(newImg, frame_gray, CV_RGB2GRAY); 

	for(int i = 0; i < numFrame; ++i) 
		cvAbsDiff(frame_gray, chain[i], diff[i]); 

	frame_result = cvCreateImage(size, IPL_DEPTH_8U, 1); 
	for(int i = 0; i < frame_gray->width; ++i) 
	{ 
		for(int j = 0; j < frame_gray->height; ++j) 
		{     
			if ( mask && (mask->imageData + mask->widthStep * j)[i]==0 )
				continue; // Not under the mask
				
			double temp = 0; 
			for(int k = 0; k < numFrame; ++k) 
			{ 
				(frame_result->imageData + frame_result->widthStep * j)[i] = 1; 
				//(frame_result->imageData + frame_result->widthStep * j)[i] = 255; 
				int value = (diff[k]->imageData + diff[k]->widthStep * j)[i]; 
				temp += hash[value]; 
				/*temp += gaussian_value(value);*/ 
				if(temp > threshold) 
				{ 
					(frame_result->imageData + frame_result->widthStep * j)[i] = 0; 
					break; 
				} 
			} 
		} 
	} 


	cvSetZero(frame_diff); 
	for(int i = 0; i < numFrame; ++i) 
	{ 
		cvDiv(diff[i], dividend, phi[i], 1); 
		cvAdd(phi[i], frame_diff, frame_diff); 
	} 

	for(int k = 0; k < numMRF; ++k) 
	{ 
		for(int i = 1; i < size.width - 1; ++i) 
			for(int j = 1; j < size.height - 1; ++j) 
				(frame_MRF->imageData + frame_MRF->widthStep * j)[i] = 
					(frame_result->imageData + frame_result->widthStep * (j - 1))[i] + (frame_result->imageData + 
							frame_result->widthStep * (j + 1))[i] + (frame_result->imageData + frame_result->widthStep * j)[i 
					+ 1] + (frame_result->imageData + frame_result->widthStep * j)[i - 1]; 


		//cvAdd(frame_left_shift, frame_right_shift, frame_MRF); 
		//cvAdd(frame_MRF, frame_up_shift, frame_MRF); 
		//cvAdd(frame_MRF, frame_down_shift, frame_MRF); 

		//cvSetZero(frame_diff); 
		//for(int i = 0; i < numFrame; ++i) 
		//{ 
		//  cvDiv(diff[i], dividend, phi[i], 1); 
		//  cvAdd(phi[i], frame_diff, frame_diff); 
		//} 

		cvSetZero(frame_result); 
		for(int i = 0; i < size.width; ++i) 
			for(int j = 0; j < size.height; ++j) 
				if(pow((double)(frame_diff->imageData + frame_diff->widthStep * j)[i], 
							2) > 2 * pow(segama, 2) * (2.5 -  2 * (frame_MRF->imageData + frame_MRF->widthStep * (j + 1))[i + 1])) 
					if(k != numMRF - 1)
						(frame_result->imageData + frame_result->widthStep * j)[i] = 1;
					else
						(frame_result->imageData + frame_result->widthStep * j)[i] = 255;
	} 
	//cvDilate(frame_result, frame_result); 
	//cvErode(frame_result, frame_result); 
	//cvShowImage("origin", frame); 
	/*cvShowImage("test", frame_left_shift);*/ 
	//cvShowImage("result", frame_result); 


	for(int i = 0; i < frame_gray->width; ++i) 
	{ 
		for(int j = 0; j < frame_gray->height; ++j) 
		{ 
			if((frame_result->imageData + frame_result->widthStep * j)[i] == 0) 
			{ 
				(chain[frameIndex%numFrame]->imageData + chain[frameIndex%numFrame]->widthStep * j)[i] 
					= (frame_gray->imageData + frame_gray->widthStep * j)[i]; 
			} 
			else 
			{ 
				(chain[frameIndex%numFrame]->imageData + chain[frameIndex%numFrame]->widthStep * j)[i] 
					= (int)((1 - alpha) * (double)(chain[frameIndex%numFrame]->imageData + chain[frameIndex%numFrame]->widthStep * 
								j)[i] + alpha * (double)(frame_gray->imageData + frame_gray->widthStep * j)[i]); 
			} 
		}
	}

	TimeEnd = clock(); 

	//cout << "The KDE time is: " << TimeEnd - TimeStart << endl; 

	assert(pFgImg != NULL);
	cvCopy(frame_result, pFgImg);
	//return 1; 
	return TimeEnd - TimeStart;

}
