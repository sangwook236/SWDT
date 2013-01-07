//#include "stdafx.h"
#include <mrpt/core.h>


namespace {
namespace local {

void extract_features(const mrpt::utils::CMRPTImage &img, const unsigned int mode, mrpt::vision::CFeatureList &features)
{
	//mrpt::vision::TImageROI ROI(200, 400, 0, img.getHeight() - 1);
	mrpt::utils::CTicTac tictac;

	if ((0x01 & mode) == 0x01)
	{
		std::cout << "extracting Harris features... [feature_harris.txt]" << std::endl;
		tictac.Tic();

		mrpt::vision::CFeatureExtraction featureExtracter;
		featureExtracter.options.featsType = mrpt::vision::featHarris;
		//featureExtracter.options.patchSize = ;
		//featureExtracter.options.harrisOptions.threshold = ;
		//featureExtracter.options.harrisOptions.k = ;
		//featureExtracter.options.harrisOptions.sigma = ;
		//featureExtracter.options.harrisOptions.radius = ;
		//featureExtracter.options.harrisOptions.min_distance = ;

		featureExtracter.detectFeatures(img, features);

		const double elapsedTime = tictac.Tac() * 1000;
		std::cout << "detected " << features.size() << " features in " << elapsedTime << std::endl;

		features.saveToTextFile(".\\robotics_data\\mrpt\\feature\\feature_harris.txt");
	}

	if ((0x02 & mode) == 0x02)
	{
		std::cout << "computing SIFT descriptors only ... [feature_harris+sift.txt]" << std::endl;
		tictac.Tic();

		mrpt::vision::CFeatureExtraction featureExtracter;
		featureExtracter.options.featsType = mrpt::vision::featHarris;
		//featureExtracter.options.patchSize = ;
		featureExtracter.options.SIFTOptions.implementation = mrpt::vision::CFeatureExtraction::Hess;

		featureExtracter.detectFeatures(img, features);
		featureExtracter.computeDescriptors(img, features, mrpt::vision::descSIFT);

		const double elapsedTime = tictac.Tac() * 1000;
		std::cout << "detected " << features.size() << " features in " << elapsedTime << std::endl;

		features.saveToTextFile(".\\robotics_data\\mrpt\\feature\\feature_harris+sift.txt");
	}

	if ((0x04 & mode) == 0x04)
	{
		std::cout << "Extracting KLT features... [feature_klt.txt]" << std::endl;
		tictac.Tic();

		mrpt::vision::CFeatureExtraction featureExtracter;
		featureExtracter.options.featsType = mrpt::vision::featKLT;
		//featureExtracter.options.patchSize = ;
		//featureExtracter.options.KLTOptions.radius = ;
		//featureExtracter.options.KLTOptions.threshold = ;
		//featureExtracter.options.KLTOptions.min_distance = ;

		featureExtracter.detectFeatures(img, features);

		const double elapsedTime = tictac.Tac() * 1000;
		std::cout << "detected " << features.size() << " features in " << elapsedTime << std::endl;

		features.saveToTextFile(".\\robotics_data\\mrpt\\feature\\feature_klt.txt");
	}

	if ((0x08 & mode) == 0x08)
	{
		std::cout << "extracting SIFT features... [feature_sift_hess.txt]" << std::endl;
		tictac.Tic();

		mrpt::vision::CFeatureExtraction featureExtracter;
		featureExtracter.options.featsType = mrpt::vision::featSIFT;
		//featureExtracter.options.patchSize = ;
		featureExtracter.options.SIFTOptions.implementation = mrpt::vision::CFeatureExtraction::Hess;

		featureExtracter.detectFeatures(img, features);

		const double elapsedTime = tictac.Tac() * 1000;
		std::cout << "detected " << features.size() << " features in " << elapsedTime << std::endl;

		features.saveToTextFile(".\\robotics_data\\mrpt\\feature\\feature_sift_hess.txt");
	}

	if ((0x10 & mode) == 0x10)
	{
		std::cout << "extracting SURF features... [feature_surf.txt]" << std::endl;
		tictac.Tic();

		mrpt::vision::CFeatureExtraction featureExtracter;
		featureExtracter.options.featsType = mrpt::vision::featSURF;
		//featureExtracter.options.patchSize = ;
		//featureExtracter.options.SURFOptions.rotation_invariant = ;
	
		featureExtracter.detectFeatures(img, features);

		const double elapsedTime = tictac.Tac() * 1000;
		std::cout << "detected " << features.size() << " features in " << elapsedTime << std::endl;

		features.saveToTextFile(".\\robotics_data\\mrpt\\feature\\feature_surf.txt");
	}

	if ((0x20 & mode) == 0x20)
	{
		std::cout << "computing spin images descriptors only ... [feature_harris+spinimgs.txt]";
		tictac.Tic();

		mrpt::vision::CFeatureExtraction featureExtracter;
		featureExtracter.options.featsType = mrpt::vision::featHarris;
		//featureExtracter.options.patchSize = ;
		featureExtracter.options.SpinImagesOptions.radius = 13;
		featureExtracter.options.SpinImagesOptions.hist_size_distance = 10;
		featureExtracter.options.SpinImagesOptions.hist_size_intensity = 10;

		featureExtracter.detectFeatures(img, features);
		featureExtracter.computeDescriptors(img, features, mrpt::vision::descSpinImages);

		std::cout << mrpt::format("  %.03fms", tictac.Tac() * 1000) << std::endl;

		features.saveToTextFile(".\\robotics_data\\mrpt\\feature\\feature_harris+spinimgs.txt");
	}

	if ((0x40 & mode) == 0x40)
	{
		std::cout << "computing spin images descriptors only ... [feature_harris+sift+spinimgs.txt]";
		tictac.Tic();

		mrpt::vision::CFeatureExtraction featureExtracter;
		featureExtracter.options.featsType = mrpt::vision::featHarris;
		//featureExtracter.options.patchSize = ;
		featureExtracter.options.SIFTOptions.implementation = mrpt::vision::CFeatureExtraction::Hess;
		featureExtracter.options.SpinImagesOptions.radius = 13;
		featureExtracter.options.SpinImagesOptions.hist_size_distance = 10;
		featureExtracter.options.SpinImagesOptions.hist_size_intensity = 10;

		featureExtracter.detectFeatures(img, features);
		featureExtracter.computeDescriptors(img, features, mrpt::vision::TDescriptorType(mrpt::vision::descSIFT | mrpt::vision::descSpinImages));

		std::cout << mrpt::format("  %.03fms", tictac.Tac() * 1000) << std::endl;

		features.saveToTextFile(".\\robotics_data\\mrpt\\feature\\feature_harris+sift+spinimgs.txt");
	}
}

void match_features(const mrpt::vision::CFeatureList &features1, const mrpt::vision::CFeatureList &features2, const mrpt::vision::TDescriptorType descriptorType, std::vector<unsigned int> &minDistanceFeatureIndexes)
{
	mrpt::utils::CTicTac tictac;

	tictac.Tic();
	for (unsigned int i = 0; i < features1.size(); ++i)
	{
		// compute featureDistance:
		mrpt::vector_double featureDistance(features2.size());

		if (mrpt::vision::descAny == descriptorType)
		{
			for (unsigned int j = 0; j < features2.size(); ++j)
				featureDistance[j] = features1[i]->patchCorrelationTo(*features2[j]);
		}
		else
		{
			// ignore rotations
			//features1[i]->descriptors.polarImgsNoRotation = true;

			for (unsigned int j = 0; j < features2.size(); ++j)
				featureDistance[j] = features1[i]->descriptorDistanceTo(*features2[j]);
		}

		double minDistance = 0, maxDistance = 0;
		unsigned int minDistanceIdx = 0, maxDistanceIdx = 0;
		mrpt::math::minimum_maximum(featureDistance, minDistance, maxDistance, &minDistanceIdx, &maxDistanceIdx);

		const double distStdDev = mrpt::math::stddev(featureDistance);

		minDistanceFeatureIndexes.push_back(minDistanceIdx);
	}
	std::cout << "all distances computed in " << (1000.0 * tictac.Tac()) << " ms" << std::endl;
}

void display_descriptors(const mrpt::vision::CFeatureList &features1, const mrpt::vision::CFeatureList &features2, const mrpt::vision::TDescriptorType descriptorType, const unsigned int idx1, const unsigned int idx2)
{
	switch (descriptorType)
	{
	case mrpt::vision::descAny: // use patch
	case mrpt::vision::descPolarImages:
	case mrpt::vision::descLogPolarImages:
	case mrpt::vision::descSpinImages:
		{
			mrpt::gui::CDisplayWindow win1("descriptor #1"), win2("descriptor #2");

			mrpt::utils::CMRPTImage img1, img2;
			if (mrpt::vision::descAny == descriptorType)
			{
				img1 = features1[idx1]->patch;
				img2 = features2[idx2]->patch;
			}
			else if (mrpt::vision::descPolarImages == descriptorType)
			{
				img1.setFromMatrix(features1[idx1]->descriptors.PolarImg);
				img2.setFromMatrix(features2[idx2]->descriptors.PolarImg);
			}
			else if (mrpt::vision::descLogPolarImages == descriptorType)
			{
				img1.setFromMatrix(features1[idx1]->descriptors.LogPolarImg);
				img2.setFromMatrix(features2[idx2]->descriptors.LogPolarImg);
			}
			else if (mrpt::vision::descSpinImages == descriptorType)
			{
				const mrpt::math::CMatrixFloat M1(
					features1[idx1]->descriptors.SpinImg_range_rows,
					features1[idx1]->descriptors.SpinImg.size() / features1[idx1]->descriptors.SpinImg_range_rows,
					features1[idx1]->descriptors.SpinImg
				);
				mrpt::math::CMatrixFloat M2(
					features2[idx2]->descriptors.SpinImg_range_rows,
					features2[idx2]->descriptors.SpinImg.size() / features2[idx2]->descriptors.SpinImg_range_rows,
					features2[idx2]->descriptors.SpinImg
				);
				img1.setFromMatrix(M1);
				img2.setFromMatrix(M2);
			}

			while (img1.getWidth() < 100 && img1.getHeight() < 100)
				img1.scaleImage(img1.getWidth() * 2, img1.getHeight() * 2, mrpt::utils::IMG_INTERP_NN);
			while (img2.getWidth() < 100 && img2.getHeight() < 100)
				img2.scaleImage(img2.getWidth() * 2, img2.getHeight() * 2, mrpt::utils::IMG_INTERP_NN);
			win1.showImage(img1);
			win2.showImage(img2);

			win1.waitForKey();
		}
		break;
	case mrpt::vision::descSIFT:
		{
			mrpt::gui::CDisplayWindowPlots win1("descriptor #1"), win2("descriptor #2");

			mrpt::vector_float v1, v2;
			mrpt::utils::metaprogramming::copy_container_typecasting(features1[idx1]->descriptors.SIFT, v1);
			mrpt::utils::metaprogramming::copy_container_typecasting(features2[idx2]->descriptors.SIFT, v2);
			win1.plot(v1);
			win2.plot(v2);
			win1.axis_fit();
			win2.axis_fit();

			win1.waitForKey();
		}
		break;
	case mrpt::vision::descSURF:
		{
			mrpt::gui::CDisplayWindowPlots win1("descriptor #1"), win2("descriptor #2");

			win1.plot(features1[idx1]->descriptors.SURF);
			win2.plot(features2[idx2]->descriptors.SURF);
			win1.axis_fit();
			win2.axis_fit();

			win1.waitForKey();
		}
		break;
	}
}

void display_features(const mrpt::utils::CMRPTImage &image1, const mrpt::utils::CMRPTImage &image2, const mrpt::vision::CFeatureList &features1,const mrpt::vision::CFeatureList &features2, const unsigned int featureIdx1, const unsigned int featureIdx2)
{
	mrpt::utils::CMRPTImage img1(image1);
	mrpt::utils::CMRPTImage img2(image2);

	for (unsigned int i = 0; i < features1.size(); ++i)
	{
		if (featureIdx1 == i)
		{
			img1.cross(features1[i]->x, features1[i]->y, mrpt::utils::TColor::red, '+', 7);
			img1.drawCircle(features1[i]->x, features1[i]->y, 7, mrpt::utils::TColor::blue);
		}
		else
			img1.cross(features1[i]->x, features1[i]->y, mrpt::utils::TColor::gray, '+', 3);
	}

	for (unsigned int i = 0; i < features2.size(); ++i)
	{
		if (featureIdx2 == i)
		{
			img2.cross(features2[i]->x, features2[i]->y, mrpt::utils::TColor::red, '+', 7);
			img2.drawCircle(features2[i]->x, features2[i]->y, 7, mrpt::utils::TColor::blue);
		}
		else
			img2.cross(features2[i]->x, features2[i]->y, mrpt::utils::TColor::gray, '+', 3);
	}

	mrpt::gui::CDisplayWindow win1("features #1"), win2("features #2");
	win1.showImage(img1);
	win2.showImage(img2);

	win1.waitForKey();
}

}  // namespace local
}  // unnamed namespace

namespace mrpt {

void feature_extraction_and_matching()
{
	// extract features
	if (false)
	{
		//const std::string imageFileName(".\\robotics_data\\mrpt\\feature\\test_image.jpg");
		const std::string imageFileName(".\\robotics_data\\mrpt\\feature\\feature_matching_test_1.jpg");
		//const std::string imageFileName(".\\robotics_data\\mrpt\\feature\\feature_matching_test_2.jpg");

		mrpt::utils::CMRPTImage img;
		if (!img.loadFromFile(imageFileName))
		{
			std::cerr << "image file load error !!!" << std::endl;
			return;
		}
		std::cout << "loaded test image: " << imageFileName << std::endl;

		const unsigned int mode = 0x08;
		mrpt::vision::CFeatureList features;
		local::extract_features(img, mode, features);

#if MRPT_HAS_WXWIDGETS
		{
			mrpt::gui::CDisplayWindow win;
			win.setWindowTitle("detected features");
			win.showImageAndPoints(img, features, mrpt::utils::TColor::blue);
			win.waitForKey();
		}
#endif
	}

	// match features
	if (true)
	{
		const std::string imageFileName1(".\\robotics_data\\mrpt\\feature\\feature_matching_test_1.jpg");
		const std::string imageFileName2(".\\robotics_data\\mrpt\\feature\\feature_matching_test_2.jpg");
/*
		{
			mrpt::utils::CMRPTImage img1, img2;
			mrpt::utils::CMemoryStream buf1, buf2;
			buf1.assignMemoryNotOwn(sample_image1, sizeof(sample_image1));
			buf1 >> img1;
			buf2.assignMemoryNotOwn(sample_image2, sizeof(sample_image2));
			buf2 >> img2;

			std::cout << "writing image files ... ";
			img1.saveToFile(imageFileName1);
			img2.saveToFile(imageFileName2);
			std::cout << "done." << std::endl;
		}
*/
		mrpt::utils::CMRPTImage img1, img2;
		std::cout << "loading image files ... ";
		img1.loadFromFile(imageFileName1);
		img2.loadFromFile(imageFileName2);
		std::cout << "done." << std::endl;

		const unsigned int mode = 0x08;
		mrpt::vision::CFeatureList features1, features2;
		local::extract_features(img1, mode, features1);
		local::extract_features(img2, mode, features2);

		mrpt::vision::TDescriptorType descriptorType;
		switch (mode)
		{
		case 0x01:  // Harris
		case 0x04:  // KLT
			descriptorType = mrpt::vision::descAny;
			break;
		case 0x02:  // Harris + SIFT
		case 0x08:  // SIFT
			descriptorType = mrpt::vision::descSIFT;
			break;
		case 0x10:  // SURF
			descriptorType = mrpt::vision::descSURF;
			break;
		case 0x20:  // Harris + SpingImages
			descriptorType = mrpt::vision::descSpinImages;
			//descriptorType = mrpt::vision::descPolarImages;
			//descriptorType = mrpt::vision::descLogPolarImages;
			break;
		case 0x40:  // Harris + SIFT + SpingImages
			descriptorType = mrpt::vision::descSIFT;
			//descriptorType = mrpt::vision::descSpinImages;
			break;
		}
		std::vector<unsigned int> minDistanceFeatureIndexes;
		minDistanceFeatureIndexes.reserve(features1.size());
		local::match_features(features1, features2, descriptorType, minDistanceFeatureIndexes);

		//
		const unsigned int featureIdx = 50;

		// display a descriptor in its window and the best descriptor from the other image:
		//display_descriptors(features1, features2, descriptorType, featureIdx, minDistanceFeatureIndexes[featureIdx]);

		// display features
		local::display_features(img1, img2, features1, features2, featureIdx, minDistanceFeatureIndexes[featureIdx]);
	}
}

}  // namespace mrpt
