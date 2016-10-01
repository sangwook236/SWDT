/*
 *  bg_fg_blobs.cpp
 *  OpenCVTries1
 *
 *  Created by Roy Shilkrot on 11/21/10.
 *  Copyright 2010 MIT. All rights reserved.
 *
 */

#include "bg_fg_blobs.h"

cv::Scalar refineSegments(const cv::Mat &img, const cv::Mat &mask, cv::Mat &dst, std::vector<cv::Point> &contour, std::vector<cv::Point> &second_contour, cv::Point2i &previous)
{
    //int niters = 3;
    
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    
    cv::Mat temp;
    
    //cv::dilate(mask, temp, cv::Mat(), cv::Point(-1, -1), cv::niters);
    //cv::erode(temp, temp, cv::Mat(), cv::Point(-1, -1), cv::niters * 2);
    //cv::dilate(temp, temp, cv::Mat(), cv::Point(-1, -1), cv::niters);
	cv::blur(mask, temp, Size(11, 11));
	//cv::imshow("temp", temp);
	temp = temp > 95.0;
	    
    cv::findContours(temp, contours, /*hierarchy,*/ CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	
	if (dst.data == NULL)
		dst = cv::Mat::zeros(img.size(), CV_8UC1);
	else
		dst.setTo(cv::Scalar(0));
    
    if (contours.size() == 0)
        return cv::Scalar(-1, -1);
	
    // iterate through all the top-level contours,
    // draw each connected component with its own random color
    int idx = 0, largestComp = -1, secondlargest = -1;
    double maxWArea = 0, maxJArea = 0;
    std::vector<double> justarea(contours.size());
	std::vector<double> weightedarea(contours.size());
	
	//for (; idx >= 0; idx = hierarchy[idx][0])
	for (; idx < contours.size(); ++idx)
	{
		const std::vector<cv::Point> &c = contours[idx];
		cv::Scalar _mean = cv::mean(cv::Mat(contours[idx]));
		justarea[idx] = std::fabs(cv::contourArea(cv::Mat(c)));
		weightedarea[idx] = std::fabs(cv::contourArea(cv::Mat(c))) / 
			((previous.x >- 1) ? (1.0 + cv::norm(cv::Point(_mean[0], _mean[1]) - previous)) : 1.0);  // consider distance from last blob
	}
	for (idx = 0; idx < contours.size(); ++idx)
	{
		if (weightedarea[idx] > maxWArea)
        {
            maxWArea = weightedarea[idx];
            largestComp = idx;
        }
	}
	for (idx = 0; idx < contours.size(); ++idx)
	{
		if (justarea[idx] > maxJArea && idx != largestComp)
		{
			maxJArea = justarea[idx];
			secondlargest = idx;
		}
	}
	
    cv::Scalar color(255);
	//std::cout << "largest cc " << largestComp << std::endl;
	//cv::drawContours(dst, contours, largestComp, color, CV_FILLED); //, 8, hierarchy);
	//for (idx = 0; idx < contours[largestComp].size() - 1; ++idx)
	//	cv::line(dst, contours[largestComp][idx], contours[largestComp][idx + 1], color, 2);
	
	if (largestComp >= 0) {
		int num = contours[largestComp].size();
		cv::Point *pts = &(contours[largestComp][0]);
		cv::fillPoly(dst, (const Point **)(&pts), &num, 1, color);
		
		cv::Scalar b = cv::mean(cv::Mat(contours[largestComp]));
		b[2] = justarea[largestComp];
		
		contour.clear();
		contour = contours[largestComp];
		
		second_contour.clear();
		if (secondlargest >= 0)
		{
			second_contour = contours[secondlargest];
			b[3] = maxJArea;
		}
		
		previous.x = b[0];
		previous.y = b[1];
		return b;
	}
	else
		return cv::Scalar(-1, -1);
	
}

/*
void makePointsFromMask(cv::Mat &maskm, std::vector<cv::Point2f> &points, bool _add = false) {//, Mat& out)
{
	if (!_add)
		points.clear();
	for (int y = 0; y < maskm.rows; y +=10)
	{
		uchar *ptr = maskm.ptr<uchar>(y);
		for (int x = 0; x < maskm.cols; x += 10)
		{
			if (ptr[x] > 10)
			{
				points.push_back(cv::Point2f(x, y));
				//if (out.data != NULL)
				//	cv::circle(out, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), 2);
			}
		}
	}
}

void drawPoint(cv::Mat &out, std::vector<cv::Point2f> &points, cv::Scalar color, cv::Mat *maskm = NULL)
{
	for (int i = 0; i < points.size(); ++i)
	{
		if (maskm != NULL)
			if (((uchar *)maskm->data)[i] > 0)
				cv::circle(out, points[i], 1, color, 1);
		else
			cv::circle(out, points[i], 1, color, 1);
	}
}


//this is a sample for foreground detection functions
int bgfg_main(int argc, char **argv)
{
    IplImage *tmp_frame = NULL;
    CvCapture *cap = NULL;
    bool update_bg_model = true;
	
    if (argc < 2)
        cap = cvCaptureFromCAM(0);
    else
        cap = cvCaptureFromFile(argv[1]);
    
    if (!cap)
    {
        printf("can not open camera or video file\n");
        return -1;
    }
    
    tmp_frame = cvQueryFrame(cap);
    if (!tmp_frame)
    {
        printf("can not read data from the video source\n");
        return -1;
    }
	
    cvNamedWindow("BG", 1);
    cvNamedWindow("FG", 1);
	
    CvBGStatModel *bg_model = 0;
	cv::Mat frameMat(tmp_frame);
	cv::Mat out(frameMat.size(), CV_8UC1), outC(frameMat.size(), CV_8UC3);
	cv::Mat prevImg(frameMat.size(), CV_8UC1), nextImg(frameMat.size(), CV_8UC1);
	std::vector<cv::Point2f> prevPts, nextPts;
	std::vector<uchar> statusv;
	std::vector<float> errv;
	cv::Rect cursor(frameMat.cols / 2, frameMat.rows / 2, 10, 10);
	int nmfr = 0;  // non-motion frames counter
    
    for (int fr = 1; tmp_frame; tmp_frame = cvQueryFrame(cap), ++fr)
    {
        if (!bg_model)
        {
            // create BG model
            bg_model = cvCreateGaussianBGModel(tmp_frame);
            //bg_model = cvCreateFGDStatModel(tmp_frame);
            continue;
        }
        
        double t = (double)cvGetTickCount();
        cvUpdateBGStatModel(tmp_frame, bg_model, update_bg_model ? -1 : 0);
        t = (double)cvGetTickCount() - t;
		//printf("%d. %.1f\n", fr, t / (cvGetTickFrequency() * 1000.));
		//cvShowImage("BG", bg_model->background);
		//cvShowImage("FG", bg_model->foreground);
		
		cv::Mat tmp_bg_fg(bg_model->foreground);
		
		std::vector<cv::Point> c();
		refineSegments(frameMat, tmp_bg_fg, out, c);
		
		if (fr % 5 == 0
		) {
			makePointsFromMask(out, prevPts, (fr % 25 != 0));
		}

		cv::cvtColor(frameMat, nextImg, CV_BGR2GRAY);
		//cv::imshow("prev", prevImg);
		//cv::imshow("next", nextImg);

		cv::calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, statusv, errv);
		nextImg.copyTo(prevImg);
		
		cv::Mat ptsM(prevPts), nptsM(nextPts);
		cv::Mat statusM(statusv);
		cv::Scalar means = cv::mean(ptsM-nptsM, statusM);
		
		std::cout << "average motion of largest blob: " << means[0] << "," << means[1] << std::endl;

		{
			cv::Mat _tmp;
			frameMat.copyTo(_tmp); //,out);
			cv::Point mid = cv::Point(_tmp.cols / 2, _tmp.rows / 2);
			cv::line(_tmp, mid, mid + cv::Point(means[0], 0), cv::Scalar(255, 0, 0), 5);
			cv::line(_tmp, mid, mid + cv::Point(0, means[1]), cv::Scalar(0, 255, 0), 5);
			//cv::drawPoint(_tmp, prevPts, cv::Scalar(0, 0, 255));  //,Mat::ones(1, statusv.size(), CV_8UC1));
			//cv::drawPoint(_tmp, nextPts, cv::Scalar(255, 0, 0), &statusM);
			if (std::fabs(means[0]) > 2 && std::fabs(means[0]) < 60)
			{
				cursor.x -= means[0];
				//std::stringstream ss;
				//ss << "Move right-left";
				//cv::putText(_tmp, ss.str(), cv::Point(30, 30), CV_FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 255), 2);
			}
			else if(std::fabs(means[1]) > 2 && std::fabs(means[1]) < 60)
			{
				cursor.y -= means[1];
				//std::stringstream ss;
				//ss << "Move up-down";
				//cv::putText(_tmp, ss.str(), cv::Point(50, 50), CV_FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 255, 0), 2);
			}
			else
			{
				++nmfr;
			}

			cv::rectangle(_tmp, cursor, cv::Scalar(0, 0, 255), 2);
			cv::imshow("out", _tmp);
		}
		prevPts = nextPts;

		if (nmfr % 15 == 0)
		{
			cursor.x = frameMat.cols / 2;
			cursor.y = frameMat.rows / 2;
			nmfr = 0;
		}
		
        char k = cvWaitKey(5);
        if (k == 27) break;
        if (k == ' ')
            update_bg_model = !update_bg_model;
    }
		
    cvReleaseBGStatModel(&bg_model);
    cvReleaseCapture(&cap);
	
    return 0;
}
*/
