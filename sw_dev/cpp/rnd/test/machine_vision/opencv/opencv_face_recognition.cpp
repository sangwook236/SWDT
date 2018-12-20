//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/face.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>


namespace {
namespace local {

cv::Mat norm_0_255(cv::InputArray _src)
{
	cv::Mat src = _src.getMat();

	// Create and return normalized image:
	cv::Mat dst;
	switch (src.channels())
	{
	case 1:
		cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}

	return dst;
}

void read_csv(const std::string &filename, std::vector<cv::Mat> &images, std::vector<int> &labels, const char separator = ';')
{
#if defined(__GNUC__)
	std::ifstream stream(filename.c_str());
#else
	std::ifstream stream(filename);
#endif
	if (!stream)
	{
		const std::string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(cv::Error::StsBadArg, error_message);
	}

	std::string line, path, classlabel;
	while (std::getline(stream, line))
	{
		std::stringstream liness(line);
		std::getline(liness, path, separator);
		std::getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty())
		{
			images.push_back(cv::imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

void simple_example()
{
	cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
	//cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::FisherFaceRecognizer::create();
	//cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();

	{
		// holds images and labels
		std::vector<cv::Mat> images;
		std::vector<int> labels;
		// images for first person
		images.push_back(cv::imread("person0/0.jpg", cv::IMREAD_GRAYSCALE)); labels.push_back(0);
		images.push_back(cv::imread("person0/1.jpg", cv::IMREAD_GRAYSCALE)); labels.push_back(0);
		images.push_back(cv::imread("person0/2.jpg", cv::IMREAD_GRAYSCALE)); labels.push_back(0);
		// images for second person
		images.push_back(cv::imread("person1/0.jpg", cv::IMREAD_GRAYSCALE)); labels.push_back(1);
		images.push_back(cv::imread("person1/1.jpg", cv::IMREAD_GRAYSCALE)); labels.push_back(1);
		images.push_back(cv::imread("person1/2.jpg", cv::IMREAD_GRAYSCALE)); labels.push_back(1);

		// train it on the given dataset (the face images and labels).
		model->train(images, labels);
	}

	{
		// some containers to hold new image:
		std::vector<cv::Mat> new_images;
		std::vector<int> new_labels;

		model->update(new_images, new_labels);
	}

	{
		cv::Mat img = cv::imread("person1/3.jpg", cv::IMREAD_GRAYSCALE);

		const int predicted = model->predict(img);
	}
}

// ${OPENCV_HOME}/modules/contrib/doc/facerec/src/facerec_demo.cpp.
void facerec_demo()
{
    const std::string csv_filename("../data/machine_vision/opencv/???.csv");

    //
    std::vector<cv::Mat> images;
    std::vector<int> labels;
	try
	{
        read_csv(csv_filename, images, labels);
    }
	catch (const cv::Exception &e)
	{
        std::cerr << "Error opening file \"" << csv_filename << "\". Reason: " << e.msg << std::endl;
        return;
    }

    // Quit if there are not enough images for this demo.
    if (images.size() <= 1)
	{
        const std::string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(cv::Error::StsError, error_message);
    }

    // Get the height from the first image. We'll need this later in code to reshape the images to their original size:
    const int height = images[0].rows;

    // The following lines simply get the last images from your dataset and remove it from the vector.
	// This is done, so that the training data (which we learn the cv::FaceRecognizer on) and the test data we test the model with, do not overlap.
    const cv::Mat testSample = images[images.size() - 1];
    const int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();

	//
#if 1
	cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::FisherFaceRecognizer::create();
#else
	const int num_components = 10;  // 10 components.
	const double confidence_threshold = 123.0;  // a confidence threshold.
	cv::Ptr<cv::face::FaceRecognizer> model = cv::face::createFisherFaceRecognizer(num_components, confidence_threshold);
#endif
	model->train(images, labels);

	{
#if 1
		// The following line predicts the label of a given test image.
		const int predictedLabel = model->predict(testSample);
#else
		// To get the confidence of a prediction call the model with.
		int predictedLabel = -1;
		double confidence = 0.0;
		model->predict(testSample, predictedLabel, confidence);
#endif

		const std::string result_message = cv::format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
		std::cout << result_message << std::endl;
	}

	{
		// Sometimes you'll need to get/set internal model data, which isn't exposed by the public cv::FaceRecognizer.
		// Since each cv::FaceRecognizer is derived from a cv::Algorithm, you can query the data.

		// First we'll use it to set the threshold of the FaceRecognizer to 0.0 without retraining the model.
		// This can be useful if you are evaluating the model:
		model->setThreshold(0.0);

		// Now the threshold of this model is set to 0.0.
		// A prediction now returns -1, as it's impossible to have a distance below it.
		const int predictedLabel = model->predict(testSample);
		std::cout << "Predicted class = " << predictedLabel << std::endl;
	}

    // Here is how to get the eigenvalues of this Eigenfaces model:
    const cv::Mat eigenvalues = model->getEigenValues();
    // And we can do the same to display the Eigenvectors (read Eigenfaces):
    const cv::Mat W = model->getEigenVectors();
    // From this we will display the (at most) first 10 Eigenfaces:

    for (int i = 0; i < std::min(10, W.cols); ++i)
	{
        const std::string msg = cv::format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
        std::cout << msg << std::endl;

        // Get eigenvector #i.
        const cv::Mat &ev = W.col(i).clone();

        // Reshape to original size & normalize to [0...255] for cv::imshow.
        const cv::Mat &grayscale = norm_0_255(ev.reshape(1, height));

        // Show the image & apply a Jet colormap for better sensing.
        cv::Mat cgrayscale;
        cv::applyColorMap(grayscale, cgrayscale, cv::COLORMAP_JET);
        cv::imshow(cv::format("%d", i), cgrayscale);
    }
}

// ${OPENCV_HOME}/modules/contrib/doc/facerec/src/facerec_eigenfaces.cpp.
void eigenfaces_example()
{
	const std::string csv_filename("../data/machine_vision/opencv/???.csv");
	const std::string output_folder("../data/machine_vision/opencv");

	//
	std::vector<cv::Mat> images;
	std::vector<int> labels;
	try
	{
		read_csv(csv_filename, images, labels);
	}
	catch (const cv::Exception &e)
	{
		std::cerr << "Error opening file \"" << csv_filename << "\". Reason: " << e.msg << std::endl;
		return;
	}

    // Quit if there are not enough images for this demo.
    if (images.size() <= 1)
	{
        const std::string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(cv::Error::StsError, error_message);
    }

    // Get the height from the first image. We'll need this later in code to reshape the images to their original size:
    const int height = images[0].rows;

    // The following lines simply get the last images from your dataset and remove it from the vector.
	// This is done, so that the training data (which we learn the cv::FaceRecognizer on) and the test data we test the model with, do not overlap.
    const cv::Mat testSample = images[images.size() - 1];
    const int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();

	//
#if 1
	cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
#else
	const int num_components = 10;  // 10 components.
	const double confidence_threshold = 123.0;  // a confidence threshold. if the distance to the nearest neighbor is larger than the threshold, this method returns -1.
	cv::Ptr<cv::FaceRecognizer> model = cv::createEigenFaceRecognizer(num_components, confidence_threshold);
#endif
	model->train(images, labels);

	//
#if 1
	// The following line predicts the label of a given test image.
	const int predictedLabel = model->predict(testSample);
#else
	// To get the confidence of a prediction call the model with.
	int predictedLabel = -1;
	double confidence = 0.0;
	model->predict(testSample, predictedLabel, confidence);
#endif

	const std::string result_message = cv::format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	std::cout << result_message << std::endl;

	// Here is how to get the eigenvalues of this Eigenfaces model.
	const cv::Mat &eigenvalues = model->getEigenValues();
	// And we can do the same to display the Eigenvectors (read Eigenfaces).
	const cv::Mat &W = model->getEigenVectors();
	// Get the sample mean from the training data.
	const cv::Mat &mean = model->getMean();

	// Display.
	cv::imshow("eigenface - mean", norm_0_255(mean.reshape(1, images[0].rows)));
	// Save.
	//cv::imwrite(cv::format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));

	// Display or save the Eigenfaces.
	for (int i = 0; i < std::min(10, W.cols); ++i)
	{
		const std::string msg = cv::format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
		std::cout << msg << std::endl;

		// Get eigenvector #i.
		const cv::Mat &ev = W.col(i).clone();

		// Reshape to original size & normalize to [0...255] for cv::imshow.
		const cv::Mat &grayscale = norm_0_255(ev.reshape(1, height));

		// Show the image & apply a Jet colormap for better sensing.
		const cv::Mat cgrayscale;
		cv::applyColorMap(grayscale, cgrayscale, cv::COLORMAP_JET);

		// Display.
		cv::imshow(cv::format("eigenface - eigenface_%d", i), cgrayscale);
		// Save.
		//cv::imwrite(cv::format("%s/eigenface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
	}

	// Display or save the image reconstruction at some predefined steps.
	for(int num_components = std::min(W.cols, 10); num_components < std::min(W.cols, 300); num_components += 15)
	{
		// slice the eigenvectors from the model.
		cv::Mat evs = cv::Mat(W, cv::Range::all(), cv::Range(0, num_components));
		cv::Mat projection = cv::LDA::subspaceProject(evs, mean, images[0].reshape(1, 1));
		cv::Mat reconstruction = cv::LDA::subspaceReconstruct(evs, mean, projection);

		// Normalize the result.
		reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));

		// Display.
		cv::imshow(cv::format("eigenface - reconstruction_%d", num_components), reconstruction);
		// Save.
		//cv::imwrite(cv::format("%s/eigenface_reconstruction_%d.png", output_folder.c_str(), num_components), reconstruction);
	}
}

// ${OPENCV_HOME}/modules/contrib/doc/facerec/src/facerec_fisherfaces.cpp.
void fisherfaces_example()
{
	const std::string csv_filename("../data/machine_vision/opencv/???.csv");
	const std::string output_folder("../data/machine_vision/opencv");

	//
	std::vector<cv::Mat> images;
	std::vector<int> labels;
	try
	{
		read_csv(csv_filename, images, labels);
	}
	catch (const cv::Exception &e)
	{
		std::cerr << "Error opening file \"" << csv_filename << "\". Reason: " << e.msg << std::endl;
		return;
	}

	// Quit if there are not enough images for this demo.
    if (images.size() <= 1)
	{
        const std::string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(cv::Error::StsError, error_message);
    }

    // Get the height from the first image. We'll need this later in code to reshape the images to their original size:
    const int height = images[0].rows;

    // The following lines simply get the last images from your dataset and remove it from the vector.
	// This is done, so that the training data (which we learn the cv::FaceRecognizer on) and the test data we test the model with, do not overlap.
    const cv::Mat testSample = images[images.size() - 1];
    const int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();

	//
#if 1
	cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::FisherFaceRecognizer::create();
#else
	const int num_components = 10;  // 10 components.
	//const int num_components = 0;  // use all Fisherfaces.
	const double confidence_threshold = 123.0;  // a confidence threshold. if the distance to the nearest neighbor is larger than the threshold, this method returns -1.
	cv::Ptr<cv::FaceRecognizer> model = cv::createFisherFaceRecognizer(num_components, confidence_threshold);
#endif
	model->train(images, labels);

	//
#if 1
	// The following line predicts the label of a given test image.
	const int predictedLabel = model->predict(testSample);
#else
	// To get the confidence of a prediction call the model with.
	int predictedLabel = -1;
	double confidence = 0.0;
	model->predict(testSample, predictedLabel, confidence);
#endif

	const std::string result_message = cv::format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	std::cout << result_message << std::endl;

	// Here is how to get the eigenvalues of this Eigenfaces model.
	const cv::Mat &eigenvalues = model->getEigenValues();
	// And we can do the same to display the Eigenvectors (read Eigenfaces).
	cv::Mat W = model->getEigenVectors();
	// Get the sample mean from the training data.
	cv::Mat mean = model->getMean();

	// Display.
	cv::imshow("Fisherface - mean", norm_0_255(mean.reshape(1, images[0].rows)));
	// Save.
	//cv::imwrite(cv::format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));

	// Display or save the first, at most 16 Fisherfaces.
	for (int i = 0; i < std::min(16, W.cols); ++i)
	{
		const std::string msg = cv::format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
		std::cout << msg << std::endl;

		// Get eigenvector #i.
		const cv::Mat &ev = W.col(i).clone();

		// Reshape to original size & normalize to [0...255] for cv::imshow.
		const cv::Mat &grayscale = norm_0_255(ev.reshape(1, height));

		// Show the image & apply a Bone colormap for better sensing.
		cv::Mat cgrayscale;
		applyColorMap(grayscale, cgrayscale, cv::COLORMAP_BONE);

		// Display.
		cv::imshow(cv::format("Fisherface - fisherface_%d", i), cgrayscale);
		// Save.
		//cv::imwrite(cv::format("%s/fisherface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
	}

	// Display or save the image reconstruction at some predefined steps.
	for(int num_component = 0; num_component < std::min(16, W.cols); ++num_component)
	{
		// Slice the Fisherface from the model.
		const cv::Mat &ev = W.col(num_component);
		const cv::Mat &projection = cv::LDA::subspaceProject(ev, mean, images[0].reshape(1, 1));
		cv::Mat reconstruction = cv::LDA::subspaceReconstruct(ev, mean, projection);

		// Normalize the result.
		reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));

		// Display.
		cv::imshow(cv::format("Fisherface - reconstruction_%d", num_component), reconstruction);
		// Save.
		//cv::imwrite(cv::format("%s/fisherface_reconstruction_%d.png", output_folder.c_str(), num_component), reconstruction);
	}
}

// ${OPENCV_HOME}/modules/contrib/doc/facerec/src/facerec_lbph.cpp.
void lbph_example()
{
	const std::string csv_filename("../data/machine_vision/opencv/???.csv");
	const std::string output_folder("../data/machine_vision/opencv");

	//
	std::vector<cv::Mat> images;
	std::vector<int> labels;
	try
	{
		read_csv(csv_filename, images, labels);
	}
	catch (const cv::Exception &e)
	{
		std::cerr << "Error opening file \"" << csv_filename << "\". Reason: " << e.msg << std::endl;
		return;
	}

	// Quit if there are not enough images for this demo.
    if (images.size() <= 1)
	{
        const std::string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(cv::Error::StsError, error_message);
    }

    // Get the height from the first image. We'll need this later in code to reshape the images to their original size:
    const int height = images[0].rows;

    // The following lines simply get the last images from your dataset and remove it from the vector.
	// This is done, so that the training data (which we learn the cv::FaceRecognizer on) and the test data we test the model with, do not overlap.
    const cv::Mat testSample = images[images.size() - 1];
    const int testLabel = labels[labels.size() - 1];
    images.pop_back();
    labels.pop_back();

	//
#if 1
	cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
#else
	const int radius = 1;  // the radius used for building the Circular Local Binary Pattern.
	const int neighbors = 8;  // the number of sample points to build a Circular Local Binary Pattern from.
	const int grid_x = 8;  // the number of cells in the horizontal direction.
	const int grid_y = 8;  // the number of cells in the vertical direction.
	const double confidence_threshold = 123.0;  // a confidence threshold. if the distance to the nearest neighbor is larger than the threshold, this method returns -1.
	cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::createLBPHFaceRecognizer(radius, neighbors, grid_x, grid_y, confidence_threshold);
#endif
	model->train(images, labels);

	//
	{
#if 1
		// The following line predicts the label of a given test image.
		const int predictedLabel = model->predict(testSample);
#else
		// To get the confidence of a prediction call the model with.
		int predictedLabel = -1;
		double confidence = 0.0;
		model->predict(testSample, predictedLabel, confidence);
#endif

		const std::string result_message = cv::format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
		std::cout << result_message << std::endl;
	}

	{
		model->setThreshold(0.0);

		const int predictedLabel = model->predict(testSample);
		std::cout << "Predicted class = " << predictedLabel << std::endl;
	}

	//
	std::cout << "Model Information:" << std::endl;
	const std::string model_info = cv::format(
		"\tLBPH(radius=%i, neighbors=%i, grid_x=%i, grid_y=%i, threshold=%.2f)",
		model->getRadius(),
		model->getNeighbors(),
		model->getGridX(),
		model->getGridY(),
		model->getThreshold()
	);
	std::cout << model_info << std::endl;

	// We could get the histograms for example.
	std::vector<cv::Mat> histograms = model->getHistograms();
	// But should I really visualize it? Probably the length is interesting.
	std::cout << "Size of the histograms: " << histograms[0].total() << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void face_recognition()
{
	local::simple_example();  // to be tested/verified.

	local::facerec_demo();  // to be tested/verified.
	// Eigenface.
	local::eigenfaces_example();  // to be tested/verified.
	// Fisherface.
	local::fisherfaces_example();  // to be tested/verified.
	// Local binary patterns histograms (LBPH).
	local::lbph_example();  // to be tested/verified.

    cv::waitKey(0);
	cv::destroyAllWindows();
}

}  // namespace my_opencv
