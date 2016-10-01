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

// REF [file] >> ${PartsBasedDetector_HOME}/src/demo.cpp
void demo()
{
	//const std::string model_filename("./data/pose_estimation/PartsBasedDetector/model/Bird_9parts.xml");
	//const std::string model_filename("./data/pose_estimation/PartsBasedDetector/model/Car_9parts.xml");
	//const std::string model_filename("./data/pose_estimation/PartsBasedDetector/model/Dog_9parts.xml");
	//const std::string model_filename("./data/pose_estimation/PartsBasedDetector/model/Face_1050filters.xml");
	//const std::string model_filename("./data/pose_estimation/PartsBasedDetector/model/Face_frontal_sparse.xml");
	//const std::string model_filename("./data/pose_estimation/PartsBasedDetector/model/Face_small_146filters.xml");
	//const std::string model_filename("./data/pose_estimation/PartsBasedDetector/model/Motorbike_9parts.xml");
	const std::string model_filename("./data/pose_estimation/PartsBasedDetector/model/Person_8parts.xml");
	//const std::string model_filename("./data/pose_estimation/PartsBasedDetector/model/Person_26parts.xml");
	//const std::string model_filename("./data/pose_estimation/PartsBasedDetector/model/PersonINRIA_9parts.xml");
	//const std::string model_filename("./data/pose_estimation/PartsBasedDetector/model/Willowcoffee_5parts.xml");

	//const std::string input_filename("./data/pose_estimation/PartsBasedDetector/2007_000027.jpg");  // person(8)
	//const std::string input_filename("./data/pose_estimation/PartsBasedDetector/2007_000531.jpg");  // frontal face
	const std::string input_filename("./data/pose_estimation/PartsBasedDetector/2007_000847.jpg");  // small face, person(8)
	//const std::string input_filename("./data/pose_estimation/PartsBasedDetector/2007_003996.jpg");  // frontal face
	//const std::string input_filename("./data/pose_estimation/PartsBasedDetector/2007_004830.jpg");  // car

	const bool has_depth_file = false;
	const std::string depth_filename(".");

	// Determine the type of model to read.
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
		std::cout << "Unsupported model format: " << ext << std::endl;
		return;
	}

	std::cout << "Loading a model file ..." << std::endl;
	const bool ok = model->deserialize(model_filename);
	if (!ok)
	{
		std::cout << "Error deserializing file" << std::endl;
		return;
	}

	// Create the PartsBasedDetector and distribute the model parameters.
	PartsBasedDetector<float> pbd;
	pbd.distributeModel(*model);

	// Load the image from file.
	std::cout << "Loading an input file ..." << std::endl;
	cv::Mat im = cv::imread(input_filename);
	if (im.empty())
	{
		std::cout << "Image not found or invalid image format" << std::endl;
		return;
	}

	cv::Mat_<float> depth;
	if (has_depth_file)
	{
		std::cout << "Loading a depth file ..." << std::endl;
		depth = cv::imread(depth_filename, CV_LOAD_IMAGE_ANYDEPTH);

		// Convert the depth image from mm to m.
		depth = depth / 1000.0f;
	}

	// Detect potential candidates in the image.
	std::cout << "start detecting ..." << std::endl;
	double t = (double)cv::getTickCount();

	std::vector<Candidate> candidates;
	pbd.detect(im, depth, candidates);

	std::cout << "Detection time: " << ((double)cv::getTickCount() - t) / cv::getTickFrequency() << std::endl;
	std::cout << "Number of candidates: " << candidates.size() << std::endl;
	std::cout << "End detecting ..." << std::endl;

	// Display the best candidates.
	if (candidates.empty())
	{
		std::cout << "Fail to detect objects ..." << std::endl;
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
		std::cout << "Displaying results ..." << std::endl;
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
	// REF [paper] >> "Articulated pose estimation with flexible mixtures-of-parts", Y. Yang & D. Ramanan, CVPR, 2011.
	//	It is related to pictorial structures.

	local::demo();

	return 0;
}
