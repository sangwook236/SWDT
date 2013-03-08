#define WITH_MATLABIO 1
#define EIGEN_USE_NEW_STDVECTOR 1
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET 1

//#include "stdafx.h"
#include <PartsBasedDetector.hpp>
#include <Candidate.hpp>
#include <FileStorageModel.hpp>
#ifdef WITH_MATLABIO
#include <MatlabIOModel.hpp>
#endif
#include <Visualize.hpp>
#include <types.hpp>
#include <nms.hpp>
#include <Rect3.hpp>
#include <DistanceTransform.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <string>


namespace {
namespace local {

// [ref] ${PartsBasedDetector_HOME}/src/demo.cpp
void demo()
{
	//const std::string model_filename("./pose_estimation_data/PartsBasedDetector/model/Bird_9parts.xml");
	//const std::string model_filename("./pose_estimation_data/PartsBasedDetector/model/Car_9parts.xml");
	//const std::string model_filename("./pose_estimation_data/PartsBasedDetector/model/Dog_9parts.xml");
	//const std::string model_filename("./pose_estimation_data/PartsBasedDetector/model/Face_1050filters.xml");
	//const std::string model_filename("./pose_estimation_data/PartsBasedDetector/model/Face_frontal_sparse.xml");
	//const std::string model_filename("./pose_estimation_data/PartsBasedDetector/model/Face_small_146filters.xml");
	//const std::string model_filename("./pose_estimation_data/PartsBasedDetector/model/Motorbike_9parts.xml");
	const std::string model_filename("./pose_estimation_data/PartsBasedDetector/model/Person_8parts.xml");
	//const std::string model_filename("./pose_estimation_data/PartsBasedDetector/model/Person_26parts.xml");
	//const std::string model_filename("./pose_estimation_data/PartsBasedDetector/model/PersonINRIA_9parts.xml");
	//const std::string model_filename("./pose_estimation_data/PartsBasedDetector/model/Willowcoffee_5parts.xml");

	//const std::string input_filename("./pose_estimation_data/PartsBasedDetector/2007_000027.jpg");  // person(8)
	//const std::string input_filename("./pose_estimation_data/PartsBasedDetector/2007_000531.jpg");  // frontal face
	const std::string input_filename("./pose_estimation_data/PartsBasedDetector/2007_000847.jpg");  // small face, person(8)
	//const std::string input_filename("./pose_estimation_data/PartsBasedDetector/2007_003996.jpg");  // frontal face
	//const std::string input_filename("./pose_estimation_data/PartsBasedDetector/2007_004830.jpg");  // car

	const bool has_depth_file = false;
	const std::string depth_filename(".");

	// determine the type of model to read
	boost::scoped_ptr<Model> model;
	const std::string ext = boost::filesystem::path(model_filename).extension().string();
	if (ext.compare(".xml") == 0 || ext.compare(".yaml") == 0)
	{
		model.reset(new FileStorageModel);
	}
#ifdef WITH_MATLABIO
	else if (ext.compare(".mat") == 0)
	{
		model.reset(new MatlabIOModel);
	}
#endif
	else
	{
		std::cout << "unsupported model format: " << ext << std::endl;
		return;
	}

	std::cout << "loading a model file ..." << std::endl;
	const bool ok = model->deserialize(model_filename);
	if (!ok)
	{
		std::cout << "error deserializing file" << std::endl;
		return;
	}

	// create the PartsBasedDetector and distribute the model parameters
	PartsBasedDetector<float> pbd;
	pbd.distributeModel(*model);

	// load the image from file
	std::cout << "loading an input file ..." << std::endl;
	cv::Mat im = cv::imread(input_filename);
	if (im.empty())
	{
		std::cout << "image not found or invalid image format" << std::endl;
		return;
	}

	cv::Mat_<float> depth;
	if (has_depth_file)
	{
		std::cout << "loading a depth file ..." << std::endl;
		depth = cv::imread(depth_filename, CV_LOAD_IMAGE_ANYDEPTH);

		// convert the depth image from mm to m
		depth = depth / 1000.0f;
	}

	// detect potential candidates in the image
	std::cout << "start detecting ..." << std::endl;
	double t = (double)cv::getTickCount();

	cv::vector<Candidate> candidates;
	pbd.detect(im, depth, candidates);

	std::cout << "detection time: " << ((double)cv::getTickCount() - t) / cv::getTickFrequency() << std::endl;
	std::cout << "number of candidates: " << candidates.size() << std::endl;
	std::cout << "end detecting ..." << std::endl;

	// display the best candidates
	if (candidates.empty())
	{
		std::cout << "fail to detect objects ..." << std::endl;
	}
	else
	{
		std::cout << "# of candidates: " << candidates.size() << std::endl;

		//SearchSpacePruning<float> ssp;
		//const float zfactor = ...;
		//ssp.filterCandidatesByDepth(parts, candidates, depth, zfactor);

		Candidate::sort(candidates);
		const float overlap = 0.2;
		Candidate::nonMaximaSuppression(im, candidates, overlap);

		std::cout << "# of selected candidates: " << candidates.size() << std::endl;

		//
		std::cout << "displaying results ..." << std::endl;
		Visualize visualize(model->name());
		cv::Mat canvas;
		visualize.candidates(im, candidates, canvas, true);
		visualize.image(canvas);

		cv::waitKey(0);
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_parts_based_detector {

}  // namespace my_parts_based_detector

int parts_based_detector_main(int argc, char *argv[])
{
	// [ref] "Articulated pose estimation with flexible mixtures-of-parts", Y. Yang & D. Ramanan, CVPR, 2011
	//	it is related to pictorial structure

	local::demo();

	return 0;
}
