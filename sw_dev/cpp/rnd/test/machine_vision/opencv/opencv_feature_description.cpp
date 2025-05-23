//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <list>
#include <vector>
#include <cassert>


namespace {
namespace local {

void simpleMatching(const cv::DescriptorMatcher &descriptorMatcher, const cv::Mat &descriptors1, const cv::Mat &descriptors2, std::vector<cv::DMatch> &matches12)
{
	descriptorMatcher.match(descriptors1, descriptors2, matches12);
}

void crossCheckMatching(const cv::DescriptorMatcher &descriptorMatcher, const cv::Mat &descriptors1, const cv::Mat &descriptors2, std::vector<cv::DMatch> &filteredMatches12, const int knn = 1)
{
	filteredMatches12.clear();
	std::vector<std::vector<cv::DMatch> > matches12, matches21;
	descriptorMatcher.knnMatch(descriptors1, descriptors2, matches12, knn);
	descriptorMatcher.knnMatch(descriptors2, descriptors1, matches21, knn);
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

void matches2points(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &keypoints1, const std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::Point2f> &matchPoints1, std::vector<cv::Point2f> &matchPoints2)
{
	const size_t &count = matches.size();
	matchPoints1.clear();
	matchPoints2.clear();
	matchPoints1.reserve(count);
	matchPoints2.reserve(count);
	for (size_t i = 0; i < count; ++i)
	{
		const cv::DMatch &match = matches[i];
		//matchPoints2.push_back(keypoints2[match.queryIdx].pt);
		//matchPoints1.push_back(keypoints1[match.trainIdx].pt);
		matchPoints2.push_back(keypoints2[match.trainIdx].pt);
		matchPoints1.push_back(keypoints1[match.queryIdx].pt);
	}
}

void sift(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2, cv::Mat &descriptors1, cv::Mat &descriptors2, std::vector<cv::DMatch> &matches)
{
	std::cout << "Computing SIFT descriptors ..." << std::endl;
	{
		double t = (double)cv::getTickCount();

		const int nfeatures = 0;
		const int nOctaveLayers = 3;
		const double contrastThreshold = 0.04;
		const double edgeThreshold = 10;
		const double sigma = 1.6;
		cv::Ptr<cv::xfeatures2d::SIFT> extractor = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);

		extractor->compute(img1, keypoints1, descriptors1);
		extractor->compute(img2, keypoints2, descriptors2);

		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		std::cout << "\tDone computing descriptors... took " << t << " seconds." << std::endl;
	}

	// Match descriptors.
	std::cout << "Matching descriptors..." << std::endl;
	{
		double t = (double)cv::getTickCount();

		const int normType = cv::NORM_L1;  // cv::NORM_L1, cv::NORM_L2, cv::NORM_HAMMING, cv::NORM_HAMMING2;
		const bool crossCheck = false;
		cv::BFMatcher descriptorMatcher(normType, crossCheck);
		//cv::FlannBasedMatcher descriptorMatcher;

		simpleMatching(descriptorMatcher, descriptors1, descriptors2, matches);
		//crossCheckMatching(descriptorMatcher, descriptors1, descriptors2, matches, 1);

		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		std::cout << "\tDone matching descriptors... took " << t << " seconds." << std::endl;
	}
}

void surf(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2, cv::Mat &descriptors1, cv::Mat &descriptors2, std::vector<cv::DMatch> &matches)
{
	std::cout << "Computing SURF descriptors ..." << std::endl;
	{
		double t = (double)cv::getTickCount();

		const double hessianThreadhold = 100;
		const int nOctaves = 4;
		const int nOctaveLayers = 3;
		const bool extended = false;
		const bool upright = false;
		cv::Ptr<cv::xfeatures2d::SURF> extractor = cv::xfeatures2d::SURF::create(hessianThreadhold, nOctaves, nOctaveLayers, extended, upright);

		extractor->compute(img1, keypoints1, descriptors1);
		extractor->compute(img2, keypoints2, descriptors2);

		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		std::cout << "\tDone computing descriptors... took " << t << " seconds." << std::endl;
	}

	// Match descriptors.
	std::cout << "Matching descriptors..." << std::endl;
	{
		double t = (double)cv::getTickCount();

		const int normType = cv::NORM_L1;  // cv::NORM_L1, cv::NORM_L2, cv::NORM_HAMMING, cv::NORM_HAMMING2;
		const bool crossCheck = false;
		cv::BFMatcher descriptorMatcher(normType, crossCheck);
		//cv::FlannBasedMatcher descriptorMatcher;

		simpleMatching(descriptorMatcher, descriptors1, descriptors2, matches);
		//crossCheckMatching(descriptorMatcher, descriptors1, descriptors2, matches, 1);

		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		std::cout << "\tDone matching descriptors... took " << t << " seconds." << std::endl;
	}
}

#if 0
void train_calonder_classifier(const cv::FeatureDetector &featureDetector, const std::string &trainImagesFilename, const std::string &classifierFilename)
{
	if (trainImagesFilename.empty())
	{
		std::cout << "Invalid train images filename." << std::endl;
		return;
	}

    // Reads train images.
	std::ifstream stream(trainImagesFilename.c_str(), std::ios::in);
	std::vector<cv::Mat> trainImgs;
    while (!stream.eof())
    {
		std::string str;
		std::getline(stream, str);
        if (str.empty()) break;

		const cv::Mat &img = cv::imread(str, cv::IMREAD_GRAYSCALE);
        if (!img.empty())
            trainImgs.push_back(img);
    }

    if (trainImgs.empty())
    {
		std::cout << "All train images can not be read." << std::endl;
        return;
    }
	std::cout << trainImgs.size() << " train images were read." << std::endl;

    // Extract keypoints from train images.
	std::vector<cv::BaseKeypoint> trainPoints;
	std::vector<IplImage> iplTrainImgs(trainImgs.size());
    for (size_t imgIdx = 0; imgIdx < trainImgs.size(); ++imgIdx)
    {
        iplTrainImgs[imgIdx] = trainImgs[imgIdx];
		std::vector<cv::KeyPoint> kps;
		featureDetector.detect(trainImgs[imgIdx], kps);

        for (size_t pointIdx = 0; pointIdx < kps.size(); ++pointIdx)
        {
			const cv::Point2f &p = kps[pointIdx].pt;
			trainPoints.push_back(cv::BaseKeypoint(cvRound(p.x), cvRound(p.y), &iplTrainImgs[imgIdx]));
        }
    }

    // Trains Calonder classifier on extracted points.
	cv::RTreeClassifier classifier;
	classifier.train(trainPoints, cv::theRNG(), 48, 9, 100);

    // Writes classifier.
    classifier.write(classifierFilename.c_str());
}
#endif

#if 0
void calonder(const std::string &classifierFilename, const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2, cv::Mat &descriptors1, cv::Mat &descriptors2, std::vector<cv::DMatch> &matches)
{
	if (classifierFilename.empty())
	{
		std::cout << "Invalid classifier filename." << std::endl;
		return;
	}

	std::cout << "Computing Calonder's descriptors ..." << std::endl;
	{
		double t = (double)cv::getTickCount();

		cv::CalonderDescriptorExtractor<float> extractor(classifierFilename);

		extractor.compute(img1, keypoints1, descriptors1);
		extractor.compute(img2, keypoints2, descriptors2);

		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		std::cout << "\tDone computing descriptors... took " << t << " seconds." << std::endl;
	}

	// Match descriptors.
	std::cout << "Matching descriptors ..." << std::endl;
	{
		double t = (double)cv::getTickCount();

		//cv::BFMatcher descriptorMatcher(cv::NORM_L1, false);
		cv::BFMatcher descriptorMatcher(cv::NORM_L2, false);
		//cv::FlannBasedMatcher descriptorMatcher;

		simpleMatching(descriptorMatcher, descriptors1, descriptors2, matches);
		//crossCheckMatching(descriptorMatcher, descriptors1, descriptors2, matches, 1);

		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		std::cout << "\tDone matching descriptors... took " << t << " seconds." << std::endl;
	}
}
#endif

void brief(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2, cv::Mat &descriptors1, cv::Mat &descriptors2, std::vector<cv::DMatch> &matches)
{
	// Compute descriptors.
	std::cout << "Computing BRIEF descriptors ..." << std::endl;
	{
		double t = (double)cv::getTickCount();

		cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(32, false);  // this is really 32 x 8 matches since they are binary matches packed into bytes

		extractor->compute(img1, keypoints1, descriptors1);
		extractor->compute(img2, keypoints2, descriptors2);

		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		std::cout << "\tDone computing descriptors... took " << t << " seconds." << std::endl;
	}

	// Match descriptors.
	std::cout << "Matching descriptors..." << std::endl;
	{
		double t = (double)cv::getTickCount();

		cv::BFMatcher descriptorMatcher(cv::NORM_HAMMING);
		//cv::BFMatcher descriptorMatcher(cv::NORM_HAMMING2);

		simpleMatching(descriptorMatcher, descriptors1, descriptors2, matches);
		//crossCheckMatching(descriptorMatcher, descriptors1, descriptors2, matches, 1);

		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		std::cout << "\tDone matching descriptors... took " << t << " seconds." << std::endl;
	}
}

#if 0
void fern(const std::string &modelFilename, const cv::Mat &img1, const cv::Mat &img2)
{
	if (modelFilename.empty())
	{
		std::cout << "Invalid model filename." << std::endl;
		return;
	}

	cv::PlanarObjectDetector detector;
	bool needToTrain = false;
	{
		cv::FileStorage fs(modelFilename, cv::FileStorage::READ);
		if (fs.isOpened())
		{
			std::cout << "Try to load model file." << std::endl;
			detector.read(fs.getFirstTopLevelNode());
		}
		else needToTrain = true;
		fs.release();
	}

	const cv::Size patchSize(32, 32);
	cv::LDetector ldetector(7, 20, 2, 2000, patchSize.width, 2);
	ldetector.setVerbose(true);

	cv::Mat object(img1.clone()), image(img2.clone());
	// FIXME [adjust] >> adjust parameters.
	const int blurKSize = 3;
	const double sigma = 2.0;
	cv::GaussianBlur(object, object, cv::Size(blurKSize, blurKSize), sigma, sigma, cv::BORDER_DEFAULT);
	cv::GaussianBlur(image, image, cv::Size(blurKSize, blurKSize), sigma, sigma, cv::BORDER_DEFAULT);

	std::vector<cv::Mat> objpyr, imgpyr;
	cv::buildPyramid(object, objpyr, ldetector.nOctaves - 1);
	cv::buildPyramid(image, imgpyr, ldetector.nOctaves - 1);

	cv::PatchGenerator patchGenerator(0, 256, 5, true, 0.8, 1.2, -CV_PI / 2.0, CV_PI / 2.0, -CV_PI / 2.0, CV_PI / 2.0);

	if (needToTrain)
	{
		std::cout << "Model file not found" << std::endl;
		std::cout << "Start to train the model" << std::endl;
		std::cout << "\tStep 1. finding the robust keypoints ..." << std::endl;
		ldetector.setVerbose(true);
		std::vector<cv::KeyPoint> objKeypoints;
		ldetector.getMostStable2D(object, objKeypoints, 100, patchGenerator);

		std::cout << "\tStep 2. training ferns-based planar object detector ..." << std::endl;
		detector.setVerbose(true);
		detector.train(objpyr, objKeypoints, patchSize.width, 100, 11, 10000, ldetector, patchGenerator);

		std::cout << "\tStep 3. saving the model to " << modelFilename << " ...";
		cv::FileStorage fs(modelFilename, cv::FileStorage::WRITE);
		if (fs.isOpened())
			detector.write(fs, "ferns_model");
		fs.release();
	}

	const std::vector<cv::KeyPoint> &objKeypoints = detector.getModelPoints();
	std::vector<cv::KeyPoint> imgKeypoints;
	ldetector(imgpyr, imgKeypoints, 300);

	// Compute & match descriptors.
	std::cout << "Computing & matching fern descriptors ..." << std::endl;
	cv::Mat H;
	std::vector<cv::Point2f> targetCornerPoints;
	std::vector<int> pairs;

	double t = (double)cv::getTickCount();

	const bool found = detector(imgpyr, imgKeypoints, H, targetCornerPoints, &pairs);

	t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
	std::cout << "\tDone matching descriptors... took " << t << " seconds." << std::endl;

	// Draw match points.
	cv::Mat img_correspondence(object.rows + image.rows, std::max(object.cols, image.cols), CV_8UC3, cv::Scalar::all(0));
#if defined(__GNUC__)
    cv::Mat ic_tmp1(img_correspondence, cv::Rect(0, 0, object.cols, object.rows));
	cv::cvtColor(object, ic_tmp1, cv::COLOR_GRAY2BGR);
    cv::Mat ic_tmp2(img_correspondence, cv::Rect(0, object.rows, image.cols, image.rows));
    cv::cvtColor(image, ic_tmp2, cv::COLOR_GRAY2BGR);
#else
	cv::cvtColor(object, img_correspondence(cv::Rect(0, 0, object.cols, object.rows)), cv::COLOR_GRAY2BGR);
    cv::cvtColor(image, img_correspondence(cv::Rect(0, object.rows, image.cols, image.rows)), cv::COLOR_GRAY2BGR);
#endif

	for (size_t i = 0; i < objKeypoints.size(); ++i)
	{
		cv::circle(img_correspondence, objKeypoints[i].pt, 2, CV_RGB(255, 0, 0), -1);
		cv::circle(img_correspondence, objKeypoints[i].pt, (1 << objKeypoints[i].octave) * 15, CV_RGB(0, 255, 0), 1);
	}
	for (size_t i = 0; i < imgKeypoints.size(); ++i)
	{
		cv::circle(img_correspondence, cv::Point2f(0.0f, (float)object.rows) + imgKeypoints[i].pt, 2, CV_RGB(255, 0, 0), -1);
		cv::circle(img_correspondence, cv::Point2f(0.0f, (float)object.rows) + imgKeypoints[i].pt, (1 << imgKeypoints[i].octave) * 15, CV_RGB(0, 255, 0), 1);
	}

	for (size_t i = 0; i < pairs.size(); i += 2)
		cv::line(img_correspondence, objKeypoints[pairs[i]].pt, cv::Point2f(0.0f, (float)object.rows)+ imgKeypoints[pairs[i+1]].pt, CV_RGB(0, 0, 255));

	if (found)
	{
		const size_t count = targetCornerPoints.size();
		for (size_t i = 0; i < count; ++i)
		{
			const cv::Point &r1 = targetCornerPoints[i % count];
			const cv::Point &r2 = targetCornerPoints[(i+1) % count];
			cv::line(img_correspondence, cv::Point(r1.x, r1.y + object.rows), cv::Point(r2.x, r2.y + object.rows), CV_RGB(255, 0, 0), 2);
		}
	}

	cv::Mat img_warped;
	cv::warpPerspective(img1, img_warped, H, img2.size());

	cv::imshow("Feature description - Correspondence", img_correspondence);
	cv::imshow("Feature description - Warped image", img_warped);

	cv::waitKey(0);

	cv::destroyAllWindows();
}
#endif

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void feature_description()
{
#if 1
	const std::string img1_name("../data/machine_vision/opencv/box.png");
	const std::string img2_name("../data/machine_vision/opencv/box_in_scene.png");

	const std::string modelFilename("../data/machine_vision/opencv/fern_model.xml.gz");
	const std::string trainImagesFilename("../data/machine_vision/opencv/calonder_train_images_filename.txt");
	const std::string classifierFilename("../data/machine_vision/opencv/calonder_classfier.txt");
#elif 0
	const std::string img1_name("../data/machine_vision/opencv/melon_target.png");
	const std::string img2_name("../data/machine_vision/opencv/melon_1.png");
	//const std::string img2_name("../data/machine_vision/opencv/melon_2.png");
	//const std::string img2_name("../data/machine_vision/opencv/melon_3.png");

	const std::string modelFilename("../data/machine_vision/opencv/fern_model.xml.gz");
	const std::string trainImagesFilename("../data/machine_vision/opencv/calonder_train_images_filename.txt");
	const std::string classifierFilename("../data/machine_vision/opencv/calonder_classfier.txt");
#endif

	//std::cout << "Reading the images ..." << std::endl;
	const cv::Mat &img1 = cv::imread(img1_name, cv::IMREAD_GRAYSCALE);
	const cv::Mat &img2 = cv::imread(img2_name, cv::IMREAD_GRAYSCALE);
	if (img1.empty() || img2.empty())
	{
		std::cout << "Failed to load image files." << std::endl;
		return;
	}

	// Extract keypoints.
	std::cout << "Extracting keypoints ..." << std::endl;
	//cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SIFT::create();
	cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SURF::create();
	//cv::Ptr<cv::Feature2D> detector = cv::FastFeatureDetector::create();
	//cv::Ptr<cv::Feature2D> detector = cv::AgastFeatureDetector::create();
	//cv::Ptr<cv::Feature2D> detector = cv::GFTTDetector::create();
	//cv::Ptr<cv::Feature2D> detector = cv::SimpleBlobDetector::create();
	//cv::Ptr<cv::Feature2D> detector = cv::KAZE::create();
	//cv::Ptr<cv::Feature2D> detector = cv::AKAZE::create();

	const size_t MAX_KEYPOINT_COUNT = 200;

	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	detector->detect(img1, keypoints1);
	//cv::KeyPointsFilter::retainBest(keypoints1, MAX_KEYPOINT_COUNT);
	detector->detect(img2, keypoints2);
	//cv::KeyPointsFilter::retainBest(keypoints2, MAX_KEYPOINT_COUNT);
	std::cout << "\tExtracted " << keypoints1.size() << " keypoints from the first image." << std::endl;
	std::cout << "\tExtracted " << keypoints2.size() << " keypoints from the second image." << std::endl;

	std::cout << "Computing & matching descriptors ..." << std::endl;
	cv::Mat descriptors1, descriptors2;
	std::vector<cv::DMatch> matches;
	{
		//local::sift(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2, matches);
		//local::surf(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2, matches);
		//local::brief(img1, img2, keypoints1, keypoints2, descriptors1, descriptors2, matches);

		//local::train_calonder_classifier(featureDetector, trainImagesFilename, classifierFilename);
		//local::calonder(classifierFilename, img1, img2, keypoints1, keypoints2, descriptors1, descriptors2, matches);
		//local::fern(modelFilename, img1, img2);
	}

	// Draw matches.
	if (!matches.empty())
	{
		std::vector<cv::Point2f> matchedPoints1, matchedPoints2;
		local::matches2points(matches, keypoints1, keypoints2, matchedPoints1, matchedPoints2);  // Extract a list of the (x,y) location of the matches.
		std::vector<unsigned char> outlier_mask;
		const double ransacReprojThreshold = 3.0;
		const cv::Mat &H = cv::findHomography(cv::Mat(matchedPoints1), cv::Mat(matchedPoints2), outlier_mask, cv::RANSAC, ransacReprojThreshold);
		//const cv::Mat &H = cv::findHomography(cv::Mat(matchedPoints1), cv::Mat(matchedPoints2), outlier_mask, cv::LMEDS, ransacReprojThreshold);

		cv::Mat img_correspondence;
		//cv::drawKeypoints(img1, keypoints1, img_keypoints, cv::Scalar::all(-1), 0);
		cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_correspondence, cv::Scalar::all(-1), cv::Scalar::all(-1), reinterpret_cast<const std::vector<char> &>(outlier_mask));
		cv::Mat img_warped;
		cv::warpPerspective(img1, img_warped, H, img2.size());

		cv::imshow("Feature description - Correspondence", img_correspondence);
		cv::imshow("Feature description - Warped image", img_warped);

		cv::waitKey(0);

		cv::destroyAllWindows();
	}
}

}  // namespace my_opencv
