/**
 * @file:   main.cpp
 * @author: Jan Hendriks (dahoc3150 [at] yahoo.com)
 * @date:   Created on 2. Dezember 2012
 * @brief:  Example program on how to train your custom HOG detecting vector
 * for use with openCV <code>hog.setSVMDetector(_descriptor)</code>;
 * 
 * For the paper regarding Histograms of Oriented Gradients (HOG), @see http://lear.inrialpes.fr/pubs/2005/DT05/
 * You can populate the positive samples dir with files from the INRIA person detection dataset, @see http://pascal.inrialpes.fr/data/human/
 * This program uses SVMlight as machine learning algorithm (@see http://svmlight.joachims.org/), but is not restricted to it
 * Tested in Ubuntu Linux 64bit 12.04 "Precise Pangolin" with openCV 2.3.1, SVMlight 6.02, g++ 4.6.3
 * and standard HOG settings, training images of size 64x128px.
 * 
 * What this program basically does:
 * 1. Read positive and negative training sample image files from specified directories
 * 2. Calculate their HOG features and keep track of their classes (pos, neg)
 * 3. Save the feature map (vector of vectors/matrix) to file system
 * 4. Read in and pass the features and their classes to a machine learning algorithm, e.g. SVMlight
 * 5. Train the machine learning algorithm using the specified parameters
 * 6. Use the calculated support vectors and SVM model to calculate a single detecting descriptor vector
 * 
 * Build by issuing:
 * g++ `pkg-config --cflags opencv` -c -g -MMD -MP -MF main.o.d -o main.o main.cpp
 * gcc -c -g `pkg-config --cflags opencv` -MMD -MP -MF svmlight/svm_learn.o.d -o svmlight/svm_learn.o svmlight/svm_learn.c
 * gcc -c -g `pkg-config --cflags opencv` -MMD -MP -MF svmlight/svm_hideo.o.d -o svmlight/svm_hideo.o svmlight/svm_hideo.c
 * gcc -c -g `pkg-config --cflags opencv` -MMD -MP -MF svmlight/svm_common.o.d -o svmlight/svm_common.o svmlight/svm_common.c
 * g++ `pkg-config --cflags opencv` -o trainhog main.o svmlight/svm_learn.o svmlight/svm_hideo.o svmlight/svm_common.o `pkg-config --libs opencv`
 * 
 * Warning:
 * Be aware that the program may consume a considerable amount of main memory, hard disk memory and time, dependent on the amount of training samples.
 * Also be aware that (esp. for 32bit systems), there are limitations for the maximum file size which may take effect when writing the features file.
 * 
 * Terms of use:
 * This program is to be used as an example and is provided on an "as-is" basis without any warranties of any kind, either express or implied.
 * Use at your own risk.
 * For used third-party software, refer to their respective terms of use and licensing.
 */

#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <dirent.h>

#include "svmlight/svmlight.h"


#if defined(max)
#undef max
#endif


namespace {
namespace local {

static std::string toLowerCase(const std::string &in)
{
    std::string t;
    for (std::string::const_iterator i = in.begin(); i != in.end(); ++i)
	{
        t += tolower(*i);
    }
    return t;
}

static void storeCursor(void)
{
    std::cout << "\033[s";
}

static void resetCursor(void)
{
    std::cout << "\033[u";
}

/**
 * Saves the given descriptor vector to a file
 * @param descriptorVector the descriptor vector to save
 * @param _vectorIndices contains indices for the corresponding vector values (e.g. descriptorVector(0)=3.5f may have index 1)
 * @param fileName
 * @TODO Use _vectorIndices to write correct indices
 */
static void saveDescriptorVectorToFile(std::vector<float> &descriptorVector, std::vector<unsigned int> &_vectorIndices, std::string fileName)
{
    std::cout << "Saving descriptor vector to file '" << fileName << "'" << std::endl;
    std::string separator = " ";  // Use blank as default separator between single features
    std::fstream File;
    float percent;
    File.open(fileName, std::ios::out);
    if (File.good() && File.is_open())
	{
        std::cout << "Saving descriptor vector features:\t";
        storeCursor();
        for (std::size_t feature = 0; feature < descriptorVector.size(); ++feature)
		{
            if ((feature % 10 == 0) || (feature == (descriptorVector.size() - 1)))
			{
                percent = ((1 + feature) * 100.0f / descriptorVector.size());
                std::cout << std::setw(4) << feature << " (" << std::setw(3) << percent << "%%)";
				std::cout.flush();
                resetCursor();
            }
            File << descriptorVector.at(feature) << separator;
        }
        std::cout << std::endl;
        File << std::endl;
        File.flush();
        File.close();
    }
}

/**
 * For unixoid systems only: Lists all files in a given directory and returns a vector of path+name in string format
 * @param dirName
 * @param fileNames found file names in specified directory
 * @param validExtensions containing the valid file extensions for collection in lower case
 * @return 
 */
static void getFilesInDirectory(const std::string &dirName, std::vector<std::string> &fileNames, const std::vector<std::string> &validExtensions)
{
    std::cout << "Opening directory " << dirName << std::endl;
    DIR *dp = opendir(dirName.c_str());
    if (NULL != dp)
	{
		struct dirent *ep;
		size_t extensionLocation;
        while ((ep = readdir(dp)))
		{
            // Ignore (sub-)directories like . , .. , .svn, etc.
            if (ep->d_type & DT_DIR)
			{
                continue;
            }
            extensionLocation = std::string(ep->d_name).find_last_of(".");  // Assume the last point marks beginning of extension like file.ext
            // Check if extension is matching the wanted ones
            std::string tempExt = toLowerCase(std::string(ep->d_name).substr(extensionLocation + 1));
            if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end())
			{
                std::cout << "Found matching data file '" << ep->d_name << "'" << std::endl;
                fileNames.push_back((std::string) dirName + ep->d_name);
            }
			else
			{
                std::cout << "Found file does not match required file type, skipping: '" << ep->d_name << "'" << std::endl;
            }
        }
        closedir(dp);
    }
	else
	{
        std::cout << "Error opening directory '" << dirName << "'!" << std::endl;
    }
}

/**
 * This is the actual calculation from the (input) image data to the HOG descriptor/feature vector using the hog.compute() function
 * @param imageFilename file path of the image file to read and calculate feature vector from
 * @param descriptorVector the returned calculated feature std::vector<float> , 
 *      I can't comprehend why openCV implementation returns std::vector<float> instead of cv::MatExpr_<float> (e.g. cv::Mat<float>)
 * @param hog cv::HOGDescriptor containin HOG settings
 */
static void calculateFeaturesFromInput(const std::string &imageFilename, std::vector<float> &featureVector, cv::HOGDescriptor &hog, const cv::Size &winStride, const cv::Size &trainingPadding)
{
    /** for imread flags from openCV documentation, 
     * @see http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#Mat cv::imread(const std::string& filename, int flags)
     * @note If you get a compile-time error complaining about following line (esp. cv::imread),
     * you either do not have a current openCV version (>2.0) 
     * or the linking order is incorrect, try g++ -o openCVHogTrainer main.cpp `pkg-config --cflags --libs opencv`
     */
    cv::Mat imageData = cv::imread(imageFilename, 0);
    if (imageData.empty())
	{
        featureVector.clear();
        std::cout << "Error: HOG image '" << imageFilename << "' is empty, features calculation skipped!" << std::endl;
        return;
    }
    // Check for mismatching dimensions
    if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height)
	{
        featureVector.clear();
        std::cout << "Error: Image '" << imageFilename << "' dimensions (" << imageData.cols << " x " << imageData.rows << ") do not match HOG window size (" << hog.winSize.width << " x " << hog.winSize.height << ")!" << std::endl;
        return;
    }
    std::vector<cv::Point> locations;
	hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
    imageData.release();  // Release the image again after features are extracted
}

//--S [] 2013/12/19: Sang-Wook Lee
static void calculateFeaturesFromInput2(const cv::Mat &imageData, std::vector<float> &featureVector, cv::HOGDescriptor &hog, const cv::Size &winStride, const cv::Size &trainingPadding)
{
    // Check for mismatching dimensions
    if (imageData.cols != hog.winSize.width || imageData.rows != hog.winSize.height)
	{
        featureVector.clear();
        std::cout << "Error: Image's dimensions (" << imageData.cols << " x " << imageData.rows << ") do not match HOG window size (" << hog.winSize.width << " x " << hog.winSize.height << ")!" << std::endl;
        return;
    }

    std::vector<cv::Point> locations;
    hog.compute(imageData, featureVector, winStride, trainingPadding, locations);
}
//--E [] 2013/12/19: Sang-Wook Lee

/**
 * Shows the detections in the image
 * @param found vector containing valid detection rectangles
 * @param imageData the image in which the detections are drawn
 */
static void showDetections(const std::vector<cv::Rect> &found, cv::Mat &imageData)
{
    std::vector<cv::Rect> found_filtered;
    size_t i, j;
    for (i = 0; i < found.size(); ++i)
	{
        cv::Rect r = found[i];
        for (j = 0; j < found.size(); ++j)
            if (j != i && (r & found[j]) == r)
                break;
        if (j == found.size())
            found_filtered.push_back(r);
    }
    for (i = 0; i < found_filtered.size(); i++)
	{
        cv::Rect r = found_filtered[i];
        cv::rectangle(imageData, r.tl(), r.br(), cv::Scalar(64, 255, 64), 3);
    }
}

bool extract_HOG_features(cv::HOGDescriptor &hog, const std::string &posSamplesDir, const std::string &negSamplesDir, const std::string &featuresFile, const cv::Size &winStride, const cv::Size &trainingPadding)
{
	// Get the files to train from somewhere
	static std::vector<std::string> positiveTrainingImages;
	static std::vector<std::string> negativeTrainingImages;
	static std::vector<std::string> validExtensions;
	validExtensions.push_back("jpg");
	validExtensions.push_back("png");
	validExtensions.push_back("ppm");

	getFilesInDirectory(posSamplesDir, positiveTrainingImages, validExtensions);
	getFilesInDirectory(negSamplesDir, negativeTrainingImages, validExtensions);
	// Retrieve the descriptor vectors from the samples
	unsigned long overallSamples = positiveTrainingImages.size() + negativeTrainingImages.size();

	// Make sure there are actually samples to train
	if (0 == overallSamples)
	{
		std::cout << "No training sample files found, nothing to do!" << std::endl;
		return EXIT_SUCCESS;
	}

	/// @WARNING: This is really important, some libraries (e.g. ROS) seems to set the system locale which takes decimal commata instead of points which causes the file input parsing to fail
	setlocale(LC_ALL, "C");  // Do not use the system locale
	setlocale(LC_NUMERIC, "C");
	setlocale(LC_ALL, "POSIX");

	std::cout << "Reading files, generating HOG features and save them to file '" << featuresFile << "':" << std::endl;;
	float percent;
	/**
	 * Save the calculated descriptor vectors to a file in a format that can be used by SVMlight for training
	 * @NOTE: If you split these steps into separate steps: 
	 * 1. calculating features into memory (e.g. into a cv::Mat or std::vector< std::vector<float> >), 
	 * 2. saving features to file / directly inject from memory to machine learning algorithm,
	 * the program may consume a considerable amount of main memory
	 */ 
	std::fstream File;
	File.open(featuresFile, std::ios::out);
	if (File.good() && File.is_open())
	{
		// Remove following line for libsvm which does not support comments
		// File << "# Use this file to train, e.g. SVMlight by issuing $ svm_learn -i 1 -a weights.txt " << featuresFile << std::endl;
		// Iterate over sample images
		for (unsigned long currentFile = 0; currentFile < overallSamples; ++currentFile)
		{
			storeCursor();
			std::vector<float> featureVector;
			// Get positive or negative sample image file path
			const std::string currentImageFile = (currentFile < positiveTrainingImages.size() ? positiveTrainingImages.at(currentFile) : negativeTrainingImages.at(currentFile - positiveTrainingImages.size()));
			// Output progress
			if (0 == (currentFile+1) % 10 || (currentFile+1) == overallSamples)
			{
				percent = ((currentFile+1) * 100.0f / overallSamples);
				std::cout << std::setw(5) << (currentFile+1) << " (" << std::setw(3) << percent << "%%):\tFile '" << currentImageFile << "'";
				std::cout.flush();
				resetCursor();
			}
			// Calculate feature vector from current image file
			//--S [] 2013/12/19: Sang-Wook Lee
			/*
			calculateFeaturesFromInput(currentImageFile, featureVector, hog, winStride, trainingPadding);
			if (!featureVector.empty())
			{
				// Put positive or negative sample class to file, 
				// true=positive, false=negative, 
				// and convert positive class to +1 and negative class to -1 for SVMlight
				File << ((currentFile < positiveTrainingImages.size()) ? "+1" : "-1");
				// Save feature vector components
				for (unsigned int feature = 0; feature < featureVector.size(); ++feature)
				{
					File << " " << (feature + 1) << ":" << featureVector.at(feature);
				}
				File << std::endl;
			}
			*/
			cv::Mat imageData = cv::imread(currentImageFile, 0);
			if (imageData.empty())
			{
				featureVector.clear();
				std::cout << "Error: HOG image '" << currentImageFile << "' is empty, features calculation skipped!" << std::endl;
				return EXIT_FAILURE;
			}
			if (imageData.cols == hog.winSize.width && imageData.rows == hog.winSize.height)
			{
				calculateFeaturesFromInput2(imageData, featureVector, hog, winStride, trainingPadding);
				if (!featureVector.empty())
				{
					// Put positive or negative sample class to file, 
					// true=positive, false=negative, 
					// and convert positive class to +1 and negative class to -1 for SVMlight
					File << ((currentFile < positiveTrainingImages.size()) ? "+1" : "-1");
					// Save feature vector components
					for (unsigned int feature = 0; feature < featureVector.size(); ++feature)
					{
						File << " " << (feature + 1) << ":" << featureVector.at(feature);
					}
					File << std::endl;
				}
			}
			else if (imageData.cols >= hog.winSize.width && imageData.rows >= hog.winSize.height)
			{
				const int possible_width = imageData.cols - hog.winSize.width + 1;
				const int possible_height = imageData.rows - hog.winSize.height + 1;
				// TODO [adapt] >> the number of sampling points in image should be adapted.
				const int nbSamplesInImage = std::max(1, (int)cvRound((double)positiveTrainingImages.size() / negativeTrainingImages.size()));
				cv::RNG &rng = cv::theRNG();
				for (int kk = 0; kk < nbSamplesInImage; ++kk)
				{
					const int start_x = rng(possible_width);
					const int start_y = rng(possible_height);
					calculateFeaturesFromInput2(imageData(cv::Rect(start_x, start_y, hog.winSize.width, hog.winSize.height)), featureVector, hog, winStride, trainingPadding);
					if (!featureVector.empty())
					{
						// Put positive or negative sample class to file, 
						// true=positive, false=negative, 
						// and convert positive class to +1 and negative class to -1 for SVMlight
						File << ((currentFile < positiveTrainingImages.size()) ? "+1" : "-1");
						// Save feature vector components
						for (unsigned int feature = 0; feature < featureVector.size(); ++feature)
						{
							File << " " << (feature + 1) << ":" << featureVector.at(feature);
						}
						File << std::endl;
					}
				}
			}
			else
			{
				featureVector.clear();
				std::cout << "Error: the size of HOG image '" << currentImageFile << "' is not proper, features calculation skipped!" << std::endl;
				return false;
			}
			//--E [] 2013/12/19: Sang-Wook Lee
		}
		std::cout << std::endl;
		File.flush();
		File.close();
	}
	else
	{
		std::cout << "Error opening file '" << featuresFile << "'!" << std::endl;
		return false;
	}

	return true;
}

bool train_SVM(SVMlight *svm, const std::string &featuresFile, const std::string &svmModelFile, const std::string &descriptorVectorFile)
{
	/// Read in and train the calculated feature vectors
	std::cout << "Calling SVMlight" << std::endl;
	svm->read_problem(const_cast<char *>(featuresFile.c_str()));
	svm->train();  // Call the core libsvm training procedure
	std::cout << "Training done, saving model file!" << std::endl;
	svm->saveModelToFile(svmModelFile);

	std::cout << "Generating representative single HOG feature vector using svmlight!" << std::endl;
	std::vector<float> descriptorVector;
	std::vector<unsigned int> descriptorVectorIndices;
	// Generate a single detecting feature vector (v1 | b) from the trained support vectors, for use e.g. with the HOG algorithm
	svm->getSingleDetectingVector(descriptorVector, descriptorVectorIndices);
	// And save the precious to file system
	saveDescriptorVectorToFile(descriptorVector, descriptorVectorIndices, descriptorVectorFile);

	return true;
}

bool test_trained_SVM_model(cv::HOGDescriptor &hog)
{
#if 0
	const int camId = -1;
	cv::VideoCapture capture(camId);
	if (!capture.isOpened())
	{
		std::cout << "Vision sensor not found: " << camId << std::endl;
		return false;
	}
#else
	const std::string video_filename("./data/feature_analysis/hog/TownCentreXVID.avi");
	cv::VideoCapture capture(video_filename);
	if (!capture.isOpened())
	{
		std::cout << "File not found: " << video_filename << std::endl;
		return false;
	}
#endif

	const double hitThreshold = 0.0;  // Threshold for the distance between features and SVM classifying plane.
	const cv::Size winStride(cv::Size(16, 16));  // Window stride. It must be a multiple of block stride.
	const cv::Size padding(cv::Size(0, 0));  // Mock parameter to keep the CPU interface compatibility. It must be (0,0).
	const double scale = 1.03;  // Coefficient of the detection window increase.
	const int groupThreshold = 2;  // Coefficient to regulate the similarity threshold. When detected, some objects can be covered by many rectangles. 0 means not to perform grouping.

	const double resize_scale = 0.25;
	cv::Mat frame, image;
	while (27 != (cv::waitKey(0) & 255))
	{
		capture >> frame;

		//cv::cvtColor(frame, image, CV_BGR2GRAY);  // If you want to work on grayscale images.
		cv::resize(frame, image, cv::Size(), resize_scale, resize_scale, cv::INTER_LINEAR);

		std::vector<cv::Rect> found;
		hog.detectMultiScale(image, found, hitThreshold, winStride, padding, scale, groupThreshold);

		showDetections(found, image);

		cv::imshow("HOG custom detection", image);
	}

	return true;
}

}  // namespace local
}  // unnamed namespace

namespace my_hog {

// [ref] https://github.com/DaHoC/trainHOG & https://github.com/DaHoC/trainHOG/wiki/trainHOG-Tutorial.
bool HOG_training_example()
{
	try
	{
		cv::theRNG();

#if 1
		// Directory containing positive sample images
		const std::string posSamplesDir = "./data/feature_analysis/hog/omega_detection/pos/";
		// Directory containing negative sample images
		const std::string negSamplesDir = "./data/feature_analysis/hog/omega_detection/neg/";
		// Set the file to write the features to
		const std::string featuresFile = "./data/feature_analysis/hog/omega_detection/genfiles/features.dat";
		// Set the file to write the SVM model to
		const std::string svmModelFile = "./data/feature_analysis/hog/omega_detection/genfiles/svmlightmodel.dat";
		// Set the file to write the resulting detecting descriptor vector to
		const std::string descriptorVectorFile = "./data/feature_analysis/hog/omega_detection/genfiles/descriptorvector.dat";
#elif 0
		// Directory containing positive sample images
		const std::string posSamplesDir = "./data/feature_analysis/hog/INRIA_Person_Dataset/Train/pos/";
		// Directory containing negative sample images
		const std::string negSamplesDir = "./data/feature_analysis/hog/INRIA_Person_Dataset/Train/neg/";
		// Set the file to write the features to
		const std::string featuresFile = "./data/feature_analysis/hog/INRIA_Person_Dataset/Train/genfiles/features.dat";
		// Set the file to write the SVM model to
		const std::string svmModelFile = "./data/feature_analysis/hog/INRIA_Person_Dataset/Train/genfiles/svmlightmodel.dat";
		// Set the file to write the resulting detecting descriptor vector to
		const std::string descriptorVectorFile = "./data/feature_analysis/hog/INRIA_Person_Dataset/Train/genfiles/descriptorvector.dat";
#endif

		cv::HOGDescriptor hog;  // Use standard parameters here.
		//--S [] 2013/12/10: Sang-Wook Lee
		//hog.winSize = cv::Size(64, 128);  // Default training images size as used in paper.
		hog.winSize = cv::Size(32, 32);
		//--E [] 2013/12/10: Sang-Wook Lee

		// HOG parameters for training that for some reason are not included in the HOG class.
		const cv::Size winStride = cv::Size(8, 8);
		const cv::Size trainingPadding = cv::Size(0, 0);

		// Extract HOG features from training dataset.
		//local::extract_HOG_features(hog, posSamplesDir, negSamplesDir, featuresFile, winStride, trainingPadding);
		// Train SVM model using SVM-Light.
		//local::train_SVM(SVMlight::getInstance(), featuresFile, svmModelFile, descriptorVectorFile);

		// Test the trained SVM model.
		{
			SVMlight::getInstance()->loadModelFromFile(svmModelFile);

			std::vector<float> descriptorVector;
			std::vector<unsigned int> descriptorVectorIndices;
			// Generate a single detecting feature vector (v1 | b) from the trained support vectors, for use e.g. with the HOG algorithm
			SVMlight::getInstance()->getSingleDetectingVector(descriptorVector, descriptorVectorIndices);
			hog.setSVMDetector(descriptorVector);  // Set our custom detecting vector.

			local::test_trained_SVM_model(hog);
		}
	}
	catch (const cv::Exception &e)
	{
		//std::cout << "OpenCV exception caught: " << e.what() << std::endl;
		//std::cout << "OpenCV exception caught: " << cvErrorStr(e.code) << std::endl;
		std::cout << "OpenCV exception caught:" << std::endl
			<< "\tdescription: " << e.err << std::endl
			<< "\tline:        " << e.line << std::endl
			<< "\tfunction:    " << e.func << std::endl
			<< "\tfile:        " << e.file << std::endl;

		return false;
	}

    return true;
}

}  // namespace my_hog
