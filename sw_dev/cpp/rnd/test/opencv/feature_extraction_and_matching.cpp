#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <iostream>
#include <list>

#define DRAW_RICH_KEYPOINTS_MODE     0
#define DRAW_OUTLIERS_MODE           0


namespace {

void simpleMatching(cv::Ptr<cv::DescriptorMatcher> &descriptorMatcher, const cv::Mat &descriptors1, const cv::Mat &descriptors2, std::vector<cv::DMatch> &matches12)
{
	descriptorMatcher->match(descriptors1, descriptors2, matches12);
}

void crossCheckMatching(cv::Ptr<cv::DescriptorMatcher> &descriptorMatcher, const cv::Mat &descriptors1, const cv::Mat &descriptors2, std::vector<cv::DMatch> &filteredMatches12, int knn = 1)
{
	filteredMatches12.clear();
	std::vector<std::vector<cv::DMatch> > matches12, matches21;
	descriptorMatcher->knnMatch(descriptors1, descriptors2, matches12, knn);
	descriptorMatcher->knnMatch(descriptors2, descriptors1, matches21, knn);
	for (size_t m = 0; m < matches12.size(); ++m)
	{
		bool findCrossCheck = false;
		for (size_t fk = 0; fk < matches12[m].size(); ++fk)
		{
			const cv::DMatch &forward = matches12[m][fk];

			for (size_t bk = 0; bk < matches21[forward.trainIdx].size(); ++bk)
			{
				const cv::DMatch &backward = matches21[forward.trainIdx][bk];
				if (backward.trainIdx == forward.queryIdx)
				{
					filteredMatches12.push_back(forward);
					findCrossCheck = true;
					break;
				}
			}

			if (findCrossCheck) break;
		}
	}
}

void warpPerspectiveRand(const cv::Mat &src, cv::Mat &dst, cv::Mat &H, cv::RNG &rng)
{
	H.create(3, 3, CV_32FC1);
	H.at<float>(0,0) = rng.uniform( 0.8f, 1.2f);
	H.at<float>(0,1) = rng.uniform(-0.1f, 0.1f);
	H.at<float>(0,2) = rng.uniform(-0.1f, 0.1f) * src.cols;
	H.at<float>(1,0) = rng.uniform(-0.1f, 0.1f);
	H.at<float>(1,1) = rng.uniform( 0.8f, 1.2f);
	H.at<float>(1,2) = rng.uniform(-0.1f, 0.1f) * src.rows;
	H.at<float>(2,0) = rng.uniform(-1e-4f, 1e-4f);
	H.at<float>(2,1) = rng.uniform(-1e-4f, 1e-4f);
	H.at<float>(2,2) = rng.uniform( 0.8f, 1.2f);

	cv::warpPerspective(src, dst, H, src.size());
}

}  // unnamed namespace

void feature_extraction_and_matching()
{
	// "FAST", "STAR", "SIFT", "SURF", "MSER", "GFTT", "HARRIS"
	// also combined format is supported: feature detector adapter name ("Grid", "Pyramid") + feature detector name (see above), e.g. "GridFAST", "PyramidSTAR", etc.
	const std::string featureDetectorName("MSER");

	// "SIFT", "SURF", "BRIEF", "Calonder" (?)
	// also combined format is supported: descriptor extractor adapter name ("Opponent") + descriptor extractor name (see above), e.g. "OpponentSIFT", etc.
	const std::string decriptorExtractorName("SIFT");

	// "BruteForce"(it uses L2), "BruteForce-L1", "BruteForce-Hamming", "BruteForce-HammingLUT", "FlannBased"
	const std::string decriptorMatcherName("FlannBased");

	cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(featureDetectorName);
	cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = cv::DescriptorExtractor::create(decriptorExtractorName);
	cv::Ptr<cv::DescriptorMatcher> descriptorMatcher = cv::DescriptorMatcher::create(decriptorMatcherName);
	//const int mactherFilterType = 0;  // none filter
	const int mactherFilterType = 1;  // cross checker filter

	const double ransacReprojThreshold = 1.0;
	const bool evaluationMode = false;

	if (detector.empty())
	{
		std::cout << "can not create detector of given types" << std::endl;
		return;
	}
	if (descriptorExtractor.empty())
	{
		std::cout << "can not create descriptor extractor of given types" << std::endl;
		return;
	}
	if (descriptorMatcher.empty())
	{
		std::cout << "can not create descriptor matcher of given types" << std::endl;
		return;
	}

	//
	const std::string filename1("opencv_data\\melon_target.png");
	const std::string filename2("opencv_data\\melon_1.png");
	//const std::string filename2("opencv_data\\melon_2.png");
	//const std::string filename2("opencv_data\\melon_3.png");

	const std::string windowName("feature extraction 2");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	//
	const cv::Mat img1 = cv::imread(filename1, CV_LOAD_IMAGE_COLOR);
	if (img1.empty())
	{
		std::cout << "fail to load image file: " << filename1 << std::endl;
		return;
	}

	// extract keypoints
	std::cout << "extracting keypoints from first image ..." << std::endl;
	std::vector<cv::KeyPoint> keypoints1;
	detector->detect(img1, keypoints1);
	std::cout << '\t' << keypoints1.size() << " points" << std::endl;

	// compute descriptors
	std::cout << "computing descriptors for keypoints from first image..." << std::endl;
	cv::Mat descriptors1;
	descriptorExtractor->compute(img1, keypoints1, descriptors1);

	cv::RNG rng = cv::theRNG();
	cv::Mat H12;
	cv::Mat img2;
	if (evaluationMode)
	{
		warpPerspectiveRand(img1, img2, H12, rng);
		if (img2.empty())
		{
			std::cout << "fail to create image" << std::endl;
			return;
		}
	}
	else
	{
		img2 = cv::imread(filename2, CV_LOAD_IMAGE_COLOR);
		if (img2.empty())
		{
			std::cout << "fail to load image file " << filename2 << std::endl;
			return;
		}
	}

	// extract keypoints
	std::cout << "extracting keypoints from second image ..." << std::endl;
	std::vector<cv::KeyPoint> keypoints2;
	detector->detect(img2, keypoints2);
	std::cout << '\t' << keypoints2.size() << " points" << std::endl;

	if (evaluationMode && !H12.empty())
	{
		std::cout << "evaluate feature detector ..." << std::endl;
		float repeatability;
		int correspCount;
		cv::evaluateFeatureDetector(img1, img2, H12, &keypoints1, &keypoints2, repeatability, correspCount);
		std::cout << "\trepeatability = " << repeatability << std::endl;
		std::cout << "\tcorrespCount = " << correspCount << std::endl;
	}

	// compute descriptors
	std::cout << "computing descriptors for keypoints from second image ..." << std::endl;
	cv::Mat descriptors2;
	descriptorExtractor->compute(img2, keypoints2, descriptors2);

	// match descriptors
	std::cout << "matching descriptors..." << std::endl;
	std::vector<cv::DMatch> filteredMatches;
	switch (mactherFilterType)
	{
	case 0:
		simpleMatching(descriptorMatcher, descriptors1, descriptors2, filteredMatches);
		break;
	case 1:
		crossCheckMatching(descriptorMatcher, descriptors1, descriptors2, filteredMatches, 1);
		break;
	}

	if (evaluationMode && !H12.empty())
	{
		std::cout << "evaluate descriptor match ..." << std::endl;
		std::vector<cv::Point2f> curve;
		cv::Ptr<cv::GenericDescriptorMatcher> gdm = new cv::VectorDescriptorMatcher(descriptorExtractor, descriptorMatcher);
		cv::evaluateGenericDescriptorMatcher(img1, img2, H12, keypoints1, keypoints2, 0, 0, curve, gdm);
		for (float l_p = 0; l_p < 1 - FLT_EPSILON; l_p += 0.1f)
			std::cout << "\t1-precision = " << l_p << "; recall = " << cv::getRecall(curve, l_p) << std::endl;
	}

	std::vector<int> queryIdxs(filteredMatches.size()), trainIdxs(filteredMatches.size());
	for (size_t i = 0; i < filteredMatches.size(); ++i)
	{
		queryIdxs[i] = filteredMatches[i].queryIdx;
		trainIdxs[i] = filteredMatches[i].trainIdx;
	}

	if (!evaluationMode && ransacReprojThreshold >= 0.0)
	{
		std::cout << "computing homography (RANSAC) ..." << std::endl;
		std::vector<cv::Point2f> points1, points2;
		cv::KeyPoint::convert(keypoints1, points1, queryIdxs);
		cv::KeyPoint::convert(keypoints2, points2, trainIdxs);

		//H12 = cv::findHomography(cv::Mat(points1), cv::Mat(points2), cv::LMEDS, ransacReprojThreshold);
		H12 = cv::findHomography(cv::Mat(points1), cv::Mat(points2), cv::RANSAC, ransacReprojThreshold);
	}

	cv::Mat drawImg;
	if (!H12.empty())  // filter outliers
	{
		std::vector<char> matchesMask(filteredMatches.size(), 0);
		std::vector<cv::Point2f> points1, points2;
		cv::KeyPoint::convert(keypoints1, points1, queryIdxs);
		cv::KeyPoint::convert(keypoints2, points2, trainIdxs);
		cv::Mat points1t;
		cv::perspectiveTransform(cv::Mat(points1), points1t, H12);

		for (size_t i1 = 0; i1 < points1.size(); ++i1)
		{
			if (cv::norm(points2[i1] - points1t.at<cv::Point2f>((int)i1, 0)) < 4)  // inlier
				matchesMask[i1] = 1;
		}

		// draw inliers
		cv::drawMatches(img1, keypoints1, img2, keypoints2, filteredMatches, drawImg, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask
#if DRAW_RICH_KEYPOINTS_MODE
			, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
#endif
		);

#if DRAW_OUTLIERS_MODE
		// draw outliers
		for (size_t i1 = 0; i1 < matchesMask.size(); ++i1)
			matchesMask[i1] = !matchesMask[i1];
		cv::drawMatches(img1, keypoints1, img2, keypoints2, filteredMatches, drawImg, CV_RGB(0, 0, 255), CV_RGB(255, 0, 0), matchesMask,
			cv::DrawMatchesFlags::DRAW_OVER_OUTIMG | cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
#endif
	}
	else
		cv::drawMatches(img1, keypoints1, img2, keypoints2, filteredMatches, drawImg);

	cv::imshow(windowName, drawImg);

	cv::waitKey(0);

	cv::destroyWindow(windowName);
}
