//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <list>

#if 0
#define DRAW_RICH_KEYPOINTS_MODE     0
#define DRAW_OUTLIERS_MODE           0
#endif


namespace {
namespace local {

void crossCheckMatching(const cv::Ptr<cv::DescriptorMatcher> &matcher, const cv::Mat &descriptors1, const cv::Mat &descriptors2, std::vector<cv::DMatch> &filteredMatches12, int knn = 1)
{
	filteredMatches12.clear();
	std::vector<std::vector<cv::DMatch> > matches12, matches21;
	matcher->knnMatch(descriptors1, descriptors2, matches12, knn);
	matcher->knnMatch(descriptors2, descriptors1, matches21, knn);
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

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

#if 0
void feature_extraction_and_matching()
{
	// "FAST", "STAR", "SIFT", "SURF", "ORB", "MSER", "GFTT", "HARRIS", "DENSE", "SimpleBlob".
	// also combined format is supported: feature detector adapter name ("Grid", "Pyramid") + feature detector name (see above), e.g. "GridFAST", "PyramidSTAR", etc.
	const std::string featureDetectorName("MSER");

	// "SIFT", "SURF", "ORB", "BRIEF", "Calonder". (?)
	// also combined format is supported: descriptor extractor adapter name ("Opponent") + descriptor extractor name (see above), e.g. "OpponentSIFT", etc.
	const std::string decriptorExtractorName("SIFT");

	// "BruteForce"(it uses L2), "BruteForce-L1", "BruteForce-Hamming", "BruteForce-HammingLUT", "FlannBased".
	const std::string decriptorMatcherName("FlannBased");

	cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(featureDetectorName);
	cv::Ptr<cv::DescriptorExtractor> descriptorExtractor = cv::DescriptorExtractor::create(decriptorExtractorName);
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(decriptorMatcherName);
	//const int mactherFilterType = 0;  // None filter.
	const int mactherFilterType = 1;  // Cross checker filter.

	const double ransacReprojThreshold = 1.0;
	const bool evaluationMode = false;

	if (detector.empty())
	{
		std::cout << "Can not create detector of given types." << std::endl;
		return;
	}
	if (descriptorExtractor.empty())
	{
		std::cout << "Can not create descriptor extractor of given types." << std::endl;
		return;
	}
	if (matcher.empty())
	{
		std::cout << "Can not create descriptor matcher of given types." << std::endl;
		return;
	}

	//
	const std::string filename1("./data/machine_vision/opencv/melon_target.png");
	const std::string filename2("./data/machine_vision/opencv/melon_1.png");
	//const std::string filename2("./data/machine_vision/opencv/melon_2.png");
	//const std::string filename2("./data/machine_vision/opencv/melon_3.png");

	const std::string windowName("Feature extraction 2");
	cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

	//
	const cv::Mat rgb1(cv::imread(filename1, cv::IMREAD_COLOR));
	if (rgb1.empty())
	{
		std::cout << "Failed to load image file: " << filename1 << std::endl;
		return;
	}

	// Extract keypoints.
	std::cout << "Extracting keypoints from first image ..." << std::endl;
	std::vector<cv::KeyPoint> keypoints1;
	detector->detect(rgb1, keypoints1);
	std::cout << '\t' << keypoints1.size() << " points" << std::endl;

	// compute descriptors.
	std::cout << "Computing descriptors for keypoints from first image..." << std::endl;
	cv::Mat descriptors1;
	descriptorExtractor->compute(rgb1, keypoints1, descriptors1);

	cv::RNG rng = cv::theRNG();
	cv::Mat H12;
	cv::Mat rgb2;
	if (evaluationMode)
	{
		local::warpPerspectiveRand(rgb1, rgb2, H12, rng);
		if (rgb2.empty())
		{
			std::cout << "Failed to create image" << std::endl;
			return;
		}
	}
	else
	{
		rgb2 = cv::imread(filename2, cv::IMREAD_COLOR);
		if (rgb2.empty())
		{
			std::cout << "Fail to load image file: " << filename2 << std::endl;
			return;
		}
	}

	// Extract keypoints.
	std::cout << "Extracting keypoints from second image ..." << std::endl;
	std::vector<cv::KeyPoint> keypoints2;
	detector->detect(rgb2, keypoints2);
	std::cout << '\t' << keypoints2.size() << " points" << std::endl;

	if (evaluationMode && !H12.empty())
	{
		std::cout << "Evaluate feature detector ..." << std::endl;
		float repeatability;
		int correspCount;
		cv::evaluateFeatureDetector(rgb1, rgb2, H12, &keypoints1, &keypoints2, repeatability, correspCount);
		std::cout << "\tRepeatability = " << repeatability << std::endl;
		std::cout << "\tCorrespCount = " << correspCount << std::endl;
	}

	// Compute descriptors.
	std::cout << "Computing descriptors for keypoints from second image ..." << std::endl;
	cv::Mat descriptors2;
	descriptorExtractor->compute(rgb2, keypoints2, descriptors2);

	// Match descriptors.
	std::cout << "Matching descriptors ..." << std::endl;
	std::vector<cv::DMatch> filteredMatches;
	//std::vector<std::vector<DMatch> > filteredMatches;
	switch (mactherFilterType)
	{
	case 0:
		{
			matcher->match(descriptors1, descriptors2, filteredMatches);
			//const int k = 5;
			//matcher->knnMatch(descriptors1, descriptors2, filteredMatches, k);
			//const float maxDistance = 5.0f;
			//matcher->radiusMatch(descriptors1, descriptors2, filteredMatches, maxDistance);
		}
		break;
	case 1:
		local::crossCheckMatching(matcher, descriptors1, descriptors2, filteredMatches, 1);
		break;
	}

	if (evaluationMode && !H12.empty())
	{
		std::cout << "Evaluate descriptor match ..." << std::endl;
		std::vector<cv::Point2f> curve;
		cv::Ptr<cv::GenericDescriptorMatcher> gdm = new cv::VectorDescriptorMatcher(descriptorExtractor, matcher);
		cv::evaluateGenericDescriptorMatcher(rgb1, rgb2, H12, keypoints1, keypoints2, 0, 0, curve, gdm);
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
		std::cout << "Computing homography (RANSAC) ..." << std::endl;
		std::vector<cv::Point2f> points1, points2;
		cv::KeyPoint::convert(keypoints1, points1, queryIdxs);
		cv::KeyPoint::convert(keypoints2, points2, trainIdxs);

		//H12 = cv::findHomography(cv::Mat(points1), cv::Mat(points2), cv::LMEDS, ransacReprojThreshold);
		H12 = cv::findHomography(cv::Mat(points1), cv::Mat(points2), cv::RANSAC, ransacReprojThreshold);
	}

	cv::Mat drawImg;
	if (!H12.empty())  // Filter outliers.
	{
		std::vector<char> matchesMask(filteredMatches.size(), 0);
		std::vector<cv::Point2f> points1, points2;
		cv::KeyPoint::convert(keypoints1, points1, queryIdxs);
		cv::KeyPoint::convert(keypoints2, points2, trainIdxs);
		cv::Mat points1_transformed;
		cv::perspectiveTransform(cv::Mat(points1), points1_transformed, H12);

		for (size_t i1 = 0; i1 < points1.size(); ++i1)
		{
			if (cv::norm(points2[i1] - points1_transformed.at<cv::Point2f>((int)i1, 0)) < 4)  // inlier
				matchesMask[i1] = 1;
		}

		// Draw inliers.
		cv::drawMatches(rgb1, keypoints1, rgb2, keypoints2, filteredMatches, drawImg, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask
#if DRAW_RICH_KEYPOINTS_MODE
			, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
#endif
		);

#if DRAW_OUTLIERS_MODE
		// Draw outliers.
		for (size_t i1 = 0; i1 < matchesMask.size(); ++i1)
			matchesMask[i1] = !matchesMask[i1];
		cv::drawMatches(rgb1, keypoints1, rgb2, keypoints2, filteredMatches, drawImg, CV_RGB(0, 0, 255), CV_RGB(255, 0, 0), matchesMask,
			cv::DrawMatchesFlags::DRAW_OVER_OUTIMG | cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
#endif
	}
	else
		cv::drawMatches(rgb1, keypoints1, rgb2, keypoints2, filteredMatches, drawImg);

	cv::imshow(windowName, drawImg);

	cv::waitKey(0);

	cv::destroyWindow(windowName);
}
#else
void feature_extraction_and_matching()
{
	std::list<std::pair<std::string, std::string> > filename_pairs;
#if 0
	filename_pairs.push_back(std::make_pair("./data/machine_vision/opencv/melon_target.png", "./data/machine_vision/opencv/melon_1.png"));
	//filename_pairs.push_back(std::make_pair("./data/machine_vision/opencv/melon_2.png", "./data/machine_vision/opencv/melon_3.png"));
#elif 1
	filename_pairs.push_back(std::make_pair("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_0.jpg", "D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_1.jpg"));
	filename_pairs.push_back(std::make_pair("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_0.jpg", "D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_2.jpg"));
	filename_pairs.push_back(std::make_pair("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_0.jpg", "D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_3.jpg"));
	filename_pairs.push_back(std::make_pair("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_0.jpg", "D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_4.jpg"));
	filename_pairs.push_back(std::make_pair("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_0.jpg", "D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_5.jpg"));
	filename_pairs.push_back(std::make_pair("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_0.jpg", "D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_6.jpg"));
	filename_pairs.push_back(std::make_pair("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_0.jpg", "D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_7.jpg"));
	filename_pairs.push_back(std::make_pair("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_0.jpg", "D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_8.jpg"));

	filename_pairs.push_back(std::make_pair("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_1.jpg", "D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_2.jpg"));
	filename_pairs.push_back(std::make_pair("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_2.jpg", "D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_3.jpg"));
	filename_pairs.push_back(std::make_pair("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_3.jpg", "D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_4.jpg"));
	filename_pairs.push_back(std::make_pair("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_4.jpg", "D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_5.jpg"));
	filename_pairs.push_back(std::make_pair("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_5.jpg", "D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_6.jpg"));
	filename_pairs.push_back(std::make_pair("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_6.jpg", "D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_7.jpg"));
	filename_pairs.push_back(std::make_pair("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_7.jpg", "D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_8.jpg"));
	filename_pairs.push_back(std::make_pair("D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_8.jpg", "D:/dataset/failure_analysis/defect/visible_ray/auto_9_view/Image_20171110/C12/resized/Review_1.jpg"));
#endif

	//
	cv::Ptr<cv::Feature2D> defaultDetector(cv::xfeatures2d::SIFT::create()), defaultDescriptor(cv::xfeatures2d::SIFT::create());

	//cv::Ptr<cv::Feature2D> detector(cv::xfeatures2d::SIFT::create()), descriptor(detector);
	//cv::Ptr<cv::Feature2D> detector(cv::xfeatures2d::SURF::create()), descriptor(detector);
	//cv::Ptr<cv::Feature2D> detector(cv::BRISK::create()), descriptor(defaultDescriptor);
	//cv::Ptr<cv::Feature2D> detector(cv::ORB::create()), descriptor(defaultDescriptor);
	//cv::Ptr<cv::Feature2D> detector(cv::MSER::create());  // Use cv::MSER::detectRegions().
	cv::Ptr<cv::Feature2D> detector(cv::FastFeatureDetector::create()), descriptor(cv::xfeatures2d::DAISY::create());
	//cv::Ptr<cv::Feature2D> detector(cv::AgastFeatureDetector::create()), descriptor(defaultDescriptor);
	//cv::Ptr<cv::Feature2D> detector(cv::GFTTDetector::create()), descriptor(defaultDescriptor);
	//cv::Ptr<cv::Feature2D> detector(cv::SimpleBlobDetector::create()), descriptor(defaultDescriptor);
	//cv::Ptr<cv::Feature2D> detector(cv::KAZE::create()), descriptor(detector);
	//cv::Ptr<cv::Feature2D> detector(cv::AKAZE::create()), descriptor(detector);
	//cv::Ptr<cv::Feature2D> descriptor(cv::xfeatures2d::FREAK::create()), detector(defaultDetector);  // Use a gray image as an input.
	//cv::Ptr<cv::Feature2D> detector(cv::xfeatures2d::StarDetector::create()), descriptor(defaultDescriptor);
	//cv::Ptr<cv::Feature2D> descriptor(cv::xfeatures2d::BriefDescriptorExtractor::create()), detector(defaultDetector);
	//cv::Ptr<cv::Feature2D> descriptor(cv::xfeatures2d::LUCID::create()), detector(defaultDetector);
	//cv::Ptr<cv::Feature2D> descriptor(cv::xfeatures2d::LATCH::create()), detector(defaultDetector);
	//cv::Ptr<cv::Feature2D> descriptor(cv::xfeatures2d::DAISY::create()), detector(defaultDetector);
	//cv::Ptr<cv::Feature2D> detector(cv::xfeatures2d::MSDDetector::create()), descriptor(defaultDescriptor);
	//cv::Ptr<cv::Feature2D> descriptor(cv::xfeatures2d::VGG::create()), detector(defaultDetector);
	//cv::Ptr<cv::Feature2D> descriptor(cv::xfeatures2d::BoostDesc::create()), detector(defaultDetector);
	//cv::Ptr<cv::Feature2D> signature(cv::xfeatures2d::PCTSignatures::create());
	//cv::Ptr<cv::Feature2D> signature(cv::xfeatures2d::PCTSignaturesSQFD::create());
	//cv::Ptr<cv::Feature2D> detector(cv::xfeatures2d::HarrisLaplaceFeatureDetector::create()), descriptor(defaultDescriptor);
	//cv::Ptr<cv::Feature2D> detector(cv::xfeatures2d::AffineFeature2D::create(detector, descriptor)), descriptor(detector);  // Affine adaptation for key points.

	//cv::Ptr<cv::DescriptorMatcher> matcher(cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED));
	cv::Ptr<cv::DescriptorMatcher> matcher(cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE));
	//cv::Ptr<cv::DescriptorMatcher> matcher(cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_SL2));
	//cv::Ptr<cv::DescriptorMatcher> matcher(cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_L1));
	//cv::Ptr<cv::DescriptorMatcher> matcher(cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING));
	//cv::Ptr<cv::DescriptorMatcher> matcher(cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMINGLUT));

	const int mactherFilterType = 0;  // None filter.
	//const int mactherFilterType = 1;  // Cross checker filter.

	const size_t MAX_KEYPOINT_COUNT = 200;
	const double ransacReprojThreshold = 3.0;

	const int kernelSize = 7;
	const double sigma = 0.0;

	cv::Mat gray1, gray2;
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	std::vector<cv::DMatch> matches;
	//std::vector<std::vector<cv::DMatch> > matches;
	cv::Mat img_matches, img_warped;
	std::vector<cv::Point2f> points1, points2;
	cv::Mat H12;
	cv::Mat points1_transformed;
	for (std::list<std::pair<std::string, std::string> >::const_iterator cit = filename_pairs.begin(); cit != filename_pairs.end(); ++cit)
	{
		const std::string filename1(cit->first);
		const std::string filename2(cit->second);

		const cv::Mat rgb1(cv::imread(filename1, cv::IMREAD_COLOR));
		if (rgb1.empty())
		{
			std::cout << "Failed to load an image file: " << filename1 << std::endl;
			continue;
		}
		const cv::Mat rgb2(cv::imread(filename2, cv::IMREAD_COLOR));
		if (rgb2.empty())
		{
			std::cout << "Failed to load an image file: " << filename2 << std::endl;
			continue;
		}

#if 1
		// Blur image.
		cv::GaussianBlur(rgb1, rgb1, cv::Size(kernelSize, kernelSize), sigma, sigma, cv::BORDER_DEFAULT);
		cv::GaussianBlur(rgb2, rgb2, cv::Size(kernelSize, kernelSize), sigma, sigma, cv::BORDER_DEFAULT);
#endif

		cv::cvtColor(rgb1, gray1, cv::COLOR_BGR2GRAY);
		cv::cvtColor(rgb2, gray2, cv::COLOR_BGR2GRAY);

		// Detect keypoints.
		std::cout << "Detecting keypoints ..." << std::endl;
		detector->detect(rgb1, keypoints1);
		std::cout << '\t' << keypoints1.size() << " points detected." << std::endl;
		cv::KeyPointsFilter::retainBest(keypoints1, MAX_KEYPOINT_COUNT);
		std::cout << '\t' << keypoints1.size() << " points filtered." << std::endl;

		detector->detect(rgb2, keypoints2);
		std::cout << '\t' << keypoints2.size() << " points detected." << std::endl;
		cv::KeyPointsFilter::retainBest(keypoints2, MAX_KEYPOINT_COUNT);
		std::cout << '\t' << keypoints2.size() << " points filtered." << std::endl;

		// Compute feature descriptors.
		std::cout << "Computing feature descriptors ..." << std::endl;
		descriptor->compute(rgb1, keypoints1, descriptors1);
		descriptor->compute(rgb2, keypoints2, descriptors2);

		// Match descriptors.
		std::cout << "Matching descriptors ..." << std::endl;
		switch (mactherFilterType)
		{
		case 0:
			{
				matcher->match(descriptors1, descriptors2, matches);
				//const int k = 5;
				//matcher->knnMatch(descriptors1, descriptors2, matches, k);
				//const float maxDistance = 5.0f;
				//matcher->radiusMatch(descriptors1, descriptors2, matches, maxDistance);
			}
			break;
		case 1:
			local::crossCheckMatching(matcher, descriptors1, descriptors2, matches, 1);
			break;
		}

		// Show results.
#if 1
		std::vector<int> queryIdxs(matches.size()), trainIdxs(matches.size());
		for (size_t i = 0; i < matches.size(); ++i)
		{
			queryIdxs[i] = matches[i].queryIdx;
			trainIdxs[i] = matches[i].trainIdx;
		}

		std::cout << "Computing homography ..." << std::endl;
		cv::KeyPoint::convert(keypoints1, points1, queryIdxs);
		cv::KeyPoint::convert(keypoints2, points2, trainIdxs);

		//H12 = cv::findHomography(cv::Mat(points1), cv::Mat(points2), 0, ransacReprojThreshold);
		//H12 = cv::findHomography(cv::Mat(points1), cv::Mat(points2), cv::LMEDS, ransacReprojThreshold);
		H12 = cv::findHomography(cv::Mat(points1), cv::Mat(points2), cv::RANSAC, ransacReprojThreshold);
		//H12 = cv::findHomography(cv::Mat(points1), cv::Mat(points2), cv::RHO, ransacReprojThreshold);

		//std::cout << "Homograph = " << H12 << std::endl;

		// Warp image.
		cv::warpPerspective(rgb1, img_warped, H12, cv::Size(300, 300), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

		// Transform points.
		std::vector<char> matchesMask(matches.size(), 0);
		cv::perspectiveTransform(cv::Mat(points1), points1_transformed, H12);

		// Draw inliers.
		for (size_t i = 0; i < points1.size(); ++i)
			if (cv::norm(points2[i] - points1_transformed.at<cv::Point2f>((int)i, 0)) < 4.0)  // Inlier.
				matchesMask[i] = 1;

		cv::drawMatches(rgb1, keypoints1, rgb2, keypoints2, matches, img_matches, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask, cv::DrawMatchesFlags::DEFAULT);
		//cv::drawMatches(rgb1, keypoints1, rgb2, keypoints2, matches, img_matches, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

#if 0
		// Draw outliers.
		for (size_t i = 0; i < matchesMask.size(); ++i)
			matchesMask[i] = !matchesMask[i];

		cv::drawMatches(rgb1, keypoints1, rgb2, keypoints2, matches, img_matches, CV_RGB(255, 0, 255), CV_RGB(255, 0, 0), matchesMask, cv::DrawMatchesFlags::DRAW_OVER_OUTIMG | cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
#endif
#else
		cv::drawMatches(rgb1, keypoints1, rgb2, keypoints2, matches, img_matches);
#endif

		cv::imshow("Feature - Match", img_matches);
		cv::imshow("Feature - Warp", img_warped);

		cv::waitKey(0);
	}

	cv::destroyAllWindows();
}
#endif

}  // namespace my_opencv
