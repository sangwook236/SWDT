//#include "stdafx.h"
#include "PascalVocDataset.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#if defined WIN32 || defined _WIN32
#include <Windows.h>
#endif
#if defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__) || defined(__linux) || defined(linux)
#include <sys/stat.h>
#endif

namespace {
namespace local {

#define DEBUG_DESC_PROGRESS

//-------------------------------------------------------------------
const std::string paramsFile = "params.xml";
const std::string vocabularyFile = "vocabulary.xml.gz";
const std::string bowImageDescriptorsDir = "/bowImageDescriptors";
const std::string svmsDir = "/svms";
const std::string plotsDir = "/plots";

//-------------------------------------------------------------------
void makeDir( const std::string& dir )
{
#if defined WIN32 || defined _WIN32
    CreateDirectoryA( dir.c_str(), 0 );
#else
    mkdir( dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH );
#endif
}

void makeUsedDirs( const std::string& rootPath )
{
    makeDir(rootPath + bowImageDescriptorsDir);
    makeDir(rootPath + svmsDir);
    makeDir(rootPath + plotsDir);
}

/****************************************************************************************\
*                            Sample on image classification                             *
\****************************************************************************************/
//
// This part of the code was a little refactor
//
struct DDMParams
{
	DDMParams() : detectorType("SURF"), descriptorType("SURF"), matcherType("BruteForce") {}
	DDMParams( const std::string _detectorType, const std::string _descriptorType, const std::string& _matcherType ) :
	detectorType(_detectorType), descriptorType(_descriptorType), matcherType(_matcherType){}
	void read( const cv::FileNode& fn )
	{
		fn["detectorType"] >> detectorType;
		fn["descriptorType"] >> descriptorType;
		fn["matcherType"] >> matcherType;
	}
	void write( cv::FileStorage& fs ) const
	{
		fs << "detectorType" << detectorType;
		fs << "descriptorType" << descriptorType;
		fs << "matcherType" << matcherType;
	}
	void print() const
	{
		std::cout << "detectorType: " << detectorType << std::endl;
		std::cout << "descriptorType: " << descriptorType << std::endl;
		std::cout << "matcherType: " << matcherType << std::endl;
	}

	std::string detectorType;
	std::string descriptorType;
	std::string matcherType;
};

//-------------------------------------------------------------------
struct VocabTrainParams
{
	VocabTrainParams() : trainObjClass("chair"), vocabSize(1000), memoryUse(200), descProportion(0.3f) {}
	VocabTrainParams( const std::string _trainObjClass, size_t _vocabSize, size_t _memoryUse, float _descProportion ) :
	trainObjClass(_trainObjClass), vocabSize(_vocabSize), memoryUse(_memoryUse), descProportion(_descProportion) {}
	void read( const cv::FileNode& fn )
	{
		fn["trainObjClass"] >> trainObjClass;
		fn["vocabSize"] >> vocabSize;
		fn["memoryUse"] >> memoryUse;
		fn["descProportion"] >> descProportion;
	}
	void write( cv::FileStorage& fs ) const
	{
		fs << "trainObjClass" << trainObjClass;
		fs << "vocabSize" << vocabSize;
		fs << "memoryUse" << memoryUse;
		fs << "descProportion" << descProportion;
	}
	void print() const
	{
		std::cout << "trainObjClass: " << trainObjClass << std::endl;
		std::cout << "vocabSize: " << vocabSize << std::endl;
		std::cout << "memoryUse: " << memoryUse << std::endl;
		std::cout << "descProportion: " << descProportion << std::endl;
	}


	std::string trainObjClass; // Object class used for training visual vocabulary.
	// It shouldn't matter which object class is specified here - visual vocab will still be the same.
	int vocabSize; //number of visual words in vocabulary to train
	int memoryUse; // Memory to preallocate (in MB) when training vocab.
	// Change this depending on the size of the dataset/available memory.
	float descProportion; // Specifies the number of descriptors to use from each image as a proportion of the total num descs.
};

//-------------------------------------------------------------------
struct SVMTrainParamsExt
{
	SVMTrainParamsExt() : descPercent(0.5f), targetRatio(0.4f), balanceClasses(true) {}
	SVMTrainParamsExt( float _descPercent, float _targetRatio, bool _balanceClasses ) :
	descPercent(_descPercent), targetRatio(_targetRatio), balanceClasses(_balanceClasses) {}
	void read( const cv::FileNode& fn )
	{
		fn["descPercent"] >> descPercent;
		fn["targetRatio"] >> targetRatio;
		fn["balanceClasses"] >> balanceClasses;
	}
	void write( cv::FileStorage& fs ) const
	{
		fs << "descPercent" << descPercent;
		fs << "targetRatio" << targetRatio;
		fs << "balanceClasses" << balanceClasses;
	}
	void print() const
	{
		std::cout << "descPercent: " << descPercent << std::endl;
		std::cout << "targetRatio: " << targetRatio << std::endl;
		std::cout << "balanceClasses: " << balanceClasses << std::endl;
	}

	float descPercent; // Percentage of extracted descriptors to use for training.
	float targetRatio; // Try to get this ratio of positive to negative samples (minimum).
	bool balanceClasses;    // Balance class weights by number of samples in each (if true cSvmTrainTargetRatio is ignored).
};

//-------------------------------------------------------------------
void readUsedParams( const cv::FileNode& fn, std::string& vocName, DDMParams& ddmParams, VocabTrainParams& vocabTrainParams, SVMTrainParamsExt& svmTrainParamsExt )
{
	fn["vocName"] >> vocName;

	cv::FileNode currFn = fn;

	currFn = fn["ddmParams"];
	ddmParams.read( currFn );

	currFn = fn["vocabTrainParams"];
	vocabTrainParams.read( currFn );

	currFn = fn["svmTrainParamsExt"];
	svmTrainParamsExt.read( currFn );
}

void writeUsedParams( cv::FileStorage& fs, const std::string& vocName, const DDMParams& ddmParams, const VocabTrainParams& vocabTrainParams, const SVMTrainParamsExt& svmTrainParamsExt )
{
	fs << "vocName" << vocName;

	fs << "ddmParams" << "{";
	ddmParams.write(fs);
	fs << "}";

	fs << "vocabTrainParams" << "{";
	vocabTrainParams.write(fs);
	fs << "}";

	fs << "svmTrainParamsExt" << "{";
	svmTrainParamsExt.write(fs);
	fs << "}";
}

void printUsedParams( const std::string& vocPath, const std::string& resDir,
	const DDMParams& ddmParams, const VocabTrainParams& vocabTrainParams,
	const SVMTrainParamsExt& svmTrainParamsExt )
{
	std::cout << "CURRENT CONFIGURATION" << std::endl;
	std::cout << "----------------------------------------------------------------" << std::endl;
	std::cout << "vocPath: " << vocPath << std::endl;
	std::cout << "resDir: " << resDir << std::endl;
	std::cout << std::endl; ddmParams.print();
	std::cout << std::endl; vocabTrainParams.print();
	std::cout << std::endl; svmTrainParamsExt.print();
	std::cout << "----------------------------------------------------------------" << std::endl << std::endl;
}

bool readVocabulary( const std::string& filename, cv::Mat& vocabulary )
{
	std::cout << "Reading vocabulary...";
	cv::FileStorage fs( filename, cv::FileStorage::READ );
	if( fs.isOpened() )
	{
		fs["vocabulary"] >> vocabulary;
		std::cout << "done" << std::endl;
		return true;
	}
	return false;
}

bool writeVocabulary( const std::string& filename, const cv::Mat& vocabulary )
{
	std::cout << "Saving vocabulary..." << std::endl;
	cv::FileStorage fs( filename, cv::FileStorage::WRITE );
	if( fs.isOpened() )
	{
		fs << "vocabulary" << vocabulary;
		return true;
	}
	return false;
}

cv::Mat trainVocabulary( const std::string& filename, PascalVocDataset& vocData, const VocabTrainParams& trainParams,
	const cv::Ptr<cv::FeatureDetector>& fdetector, const cv::Ptr<cv::DescriptorExtractor>& dextractor )
{
	cv::Mat vocabulary;
	if( !readVocabulary( filename, vocabulary) )
	{
		CV_Assert( dextractor->descriptorType() == CV_32FC1 );
		const int descByteSize = dextractor->descriptorSize()*4;
		const int maxDescCount = (trainParams.memoryUse * 1048576) / descByteSize; // Total number of descs to use for training.

		std::cout << "Extracting VOC data..." << std::endl;
		std::vector<ObdImage> images;
		std::vector<char> objectPresent;
		vocData.getClassImages( trainParams.trainObjClass, CV_OBD_TRAIN, images, objectPresent );

		std::cout << "Computing descriptors..." << std::endl;
		cv::RNG& rng = cv::theRNG();
		cv::TermCriteria terminate_criterion;
		terminate_criterion.epsilon = FLT_EPSILON;
		cv::BOWKMeansTrainer bowTrainer( trainParams.vocabSize, terminate_criterion, 3, cv::KMEANS_PP_CENTERS );

		while( images.size() > 0 )
		{
			if( bowTrainer.descripotorsCount() >= maxDescCount )
			{
				assert( bowTrainer.descripotorsCount() == maxDescCount );
#ifdef DEBUG_DESC_PROGRESS
				std::cout << "Breaking due to full memory ( descriptors count = " << bowTrainer.descripotorsCount()
					<< "; descriptor size in bytes = " << descByteSize << "; all used memory = "
					<< bowTrainer.descripotorsCount()*descByteSize << std::endl;
#endif
				break;
			}

			// Randomly pick an image from the dataset which hasn't yet been seen
			// and compute the descriptors from that image.
			int randImgIdx = rng( images.size() );
			cv::Mat colorImage = cv::imread( images[randImgIdx].path );
			std::vector<cv::KeyPoint> imageKeypoints;
			fdetector->detect( colorImage, imageKeypoints );
			cv::Mat imageDescriptors;
			dextractor->compute( colorImage, imageKeypoints, imageDescriptors );

			//check that there were descriptors calculated for the current image
			if( !imageDescriptors.empty() )
			{
				int descCount = imageDescriptors.rows;
				// Extract trainParams.descProportion descriptors from the image, breaking if the 'allDescriptors' matrix becomes full
				int descsToExtract = static_cast<int>(trainParams.descProportion * static_cast<float>(descCount));
				// Fill mask of used descriptors
				std::vector<char> usedMask( descCount, false );
				std::fill( usedMask.begin(), usedMask.begin() + descsToExtract, true );
				for( int i = 0; i < descCount; i++ )
				{
					int i1 = rng(descCount), i2 = rng(descCount);
					char tmp = usedMask[i1]; usedMask[i1] = usedMask[i2]; usedMask[i2] = tmp;
				}

				for( int i = 0; i < descCount; i++ )
				{
					if( usedMask[i] && bowTrainer.descripotorsCount() < maxDescCount )
						bowTrainer.add( imageDescriptors.row(i) );
				}
			}

#ifdef DEBUG_DESC_PROGRESS
			std::cout << images.size() << " images left, " << images[randImgIdx].id << " processed - "
				<</* descs_extracted << "/" << image_descriptors.rows << " extracted - " << */
				cvRound((static_cast<double>(bowTrainer.descripotorsCount())/static_cast<double>(maxDescCount))*100.0)
				<< " % memory used" << ( imageDescriptors.empty() ? " -> no descriptors extracted, skipping" : "") << std::endl;
#endif

			// Delete the current element from images so it is not added again
			images.erase( images.begin() + randImgIdx );
		}

		std::cout << "Maximum allowed descriptor count: " << maxDescCount << ", Actual descriptor count: " << bowTrainer.descripotorsCount() << std::endl;

		std::cout << "Training vocabulary..." << std::endl;
		vocabulary = bowTrainer.cluster();

		if( !writeVocabulary(filename, vocabulary) )
		{
			std::cout << "Error: file " << filename << " can not be opened to write" << std::endl;
			exit(-1);
		}
	}
	return vocabulary;
}

bool readBowImageDescriptor( const std::string& file, cv::Mat& bowImageDescriptor )
{
	cv::FileStorage fs( file, cv::FileStorage::READ );
	if( fs.isOpened() )
	{
		fs["imageDescriptor"] >> bowImageDescriptor;
		return true;
	}
	return false;
}

bool writeBowImageDescriptor( const std::string& file, const cv::Mat& bowImageDescriptor )
{
	cv::FileStorage fs( file, cv::FileStorage::WRITE );
	if( fs.isOpened() )
	{
		fs << "imageDescriptor" << bowImageDescriptor;
		return true;
	}
	return false;
}

// Load in the bag of words vectors for a set of images, from file if possible
void calculateImageDescriptors( const std::vector<ObdImage>& images, std::vector<cv::Mat>& imageDescriptors,
	cv::Ptr<cv::BOWImgDescriptorExtractor>& bowExtractor, const cv::Ptr<cv::FeatureDetector>& fdetector,
	const std::string& resPath )
{
	CV_Assert( !bowExtractor->getVocabulary().empty() );
	imageDescriptors.resize( images.size() );

	for( size_t i = 0; i < images.size(); i++ )
	{
		std::string filename = resPath + bowImageDescriptorsDir + "/" + images[i].id + ".xml.gz";
		if( readBowImageDescriptor( filename, imageDescriptors[i] ) )
		{
#ifdef DEBUG_DESC_PROGRESS
			std::cout << "Loaded bag of word vector for image " << i+1 << " of " << images.size() << " (" << images[i].id << ")" << std::endl;
#endif
		}
		else
		{
			cv::Mat colorImage = cv::imread( images[i].path );
#ifdef DEBUG_DESC_PROGRESS
			std::cout << "Computing descriptors for image " << i+1 << " of " << images.size() << " (" << images[i].id << ")" << std::flush;
#endif
			std::vector<cv::KeyPoint> keypoints;
			fdetector->detect( colorImage, keypoints );
#ifdef DEBUG_DESC_PROGRESS
			std::cout << " + generating BoW vector" << std::flush;
#endif
			bowExtractor->compute( colorImage, keypoints, imageDescriptors[i] );
#ifdef DEBUG_DESC_PROGRESS
			std::cout << " ...DONE " << static_cast<int>(static_cast<float>(i+1)/static_cast<float>(images.size())*100.0)
				<< " % complete" << std::endl;
#endif
			if( !imageDescriptors[i].empty() )
			{
				if( !writeBowImageDescriptor( filename, imageDescriptors[i] ) )
				{
					std::cout << "Error: file " << filename << "can not be opened to write bow image descriptor" << std::endl;
					exit(-1);
				}
			}
		}
	}
}

void removeEmptyBowImageDescriptors( std::vector<ObdImage>& images, std::vector<cv::Mat>& bowImageDescriptors,
	std::vector<char>& objectPresent )
{
	CV_Assert( !images.empty() );
	for( int i = (int)images.size() - 1; i >= 0; i-- )
	{
		bool res = bowImageDescriptors[i].empty();
		if( res )
		{
			std::cout << "Removing image " << images[i].id << " due to no descriptors..." << std::endl;
			images.erase( images.begin() + i );
			bowImageDescriptors.erase( bowImageDescriptors.begin() + i );
			objectPresent.erase( objectPresent.begin() + i );
		}
	}
}

void removeBowImageDescriptorsByCount( std::vector<ObdImage>& images, std::vector<cv::Mat> bowImageDescriptors, std::vector<char> objectPresent,
	const SVMTrainParamsExt& svmParamsExt, int descsToDelete )
{
	cv::RNG& rng = cv::theRNG();
	int pos_ex = std::count( objectPresent.begin(), objectPresent.end(), true );
	int neg_ex = std::count( objectPresent.begin(), objectPresent.end(), false );

	while( descsToDelete != 0 )
	{
		int randIdx = rng(images.size());

		// Prefer positive training examples according to svmParamsExt.targetRatio if required
		if( objectPresent[randIdx] )
		{
			if( (static_cast<float>(pos_ex)/static_cast<float>(neg_ex+pos_ex)  < svmParamsExt.targetRatio) &&
				(neg_ex > 0) && (svmParamsExt.balanceClasses == false) )
			{ continue; }
			else
			{ pos_ex--; }
		}
		else
		{ neg_ex--; }

		images.erase( images.begin() + randIdx );
		bowImageDescriptors.erase( bowImageDescriptors.begin() + randIdx );
		objectPresent.erase( objectPresent.begin() + randIdx );

		descsToDelete--;
	}
	CV_Assert( bowImageDescriptors.size() == objectPresent.size() );
}

void setSVMParams( CvSVMParams& svmParams, CvMat& class_wts_cv, const cv::Mat& responses, bool balanceClasses )
{
	int pos_ex = countNonZero(responses == 1);
	int neg_ex = countNonZero(responses == -1);
	std::cout << pos_ex << " positive training samples; " << neg_ex << " negative training samples" << std::endl;

	svmParams.svm_type = CvSVM::C_SVC;
	svmParams.kernel_type = CvSVM::RBF;
	if( balanceClasses )
	{
		cv::Mat class_wts( 2, 1, CV_32FC1 );
		// The first training sample determines the '+1' class internally, even if it is negative,
		// so store whether this is the case so that the class weights can be reversed accordingly.
		bool reversed_classes = (responses.at<float>(0) < 0.f);
		if( reversed_classes == false )
		{
			class_wts.at<float>(0) = static_cast<float>(pos_ex)/static_cast<float>(pos_ex+neg_ex); // weighting for costs of positive class + 1 (i.e. cost of false positive - larger gives greater cost)
			class_wts.at<float>(1) = static_cast<float>(neg_ex)/static_cast<float>(pos_ex+neg_ex); // weighting for costs of negative class - 1 (i.e. cost of false negative)
		}
		else
		{
			class_wts.at<float>(0) = static_cast<float>(neg_ex)/static_cast<float>(pos_ex+neg_ex);
			class_wts.at<float>(1) = static_cast<float>(pos_ex)/static_cast<float>(pos_ex+neg_ex);
		}
		class_wts_cv = class_wts;
		svmParams.class_weights = &class_wts_cv;
	}
}

void setSVMTrainAutoParams( CvParamGrid& c_grid, CvParamGrid& gamma_grid,
	CvParamGrid& p_grid, CvParamGrid& nu_grid,
	CvParamGrid& coef_grid, CvParamGrid& degree_grid )
{
	c_grid = CvSVM::get_default_grid(CvSVM::C);

	gamma_grid = CvSVM::get_default_grid(CvSVM::GAMMA);

	p_grid = CvSVM::get_default_grid(CvSVM::P);
	p_grid.step = 0;

	nu_grid = CvSVM::get_default_grid(CvSVM::NU);
	nu_grid.step = 0;

	coef_grid = CvSVM::get_default_grid(CvSVM::COEF);
	coef_grid.step = 0;

	degree_grid = CvSVM::get_default_grid(CvSVM::DEGREE);
	degree_grid.step = 0;
}

void trainSVMClassifier( CvSVM& svm, const SVMTrainParamsExt& svmParamsExt, const std::string& objClassName, PascalVocDataset& vocData,
	cv::Ptr<cv::BOWImgDescriptorExtractor>& bowExtractor, const cv::Ptr<cv::FeatureDetector>& fdetector,
	const std::string& resPath )
{
	/* first check if a previously trained svm for the current class has been saved to file */
	std::string svmFilename = resPath + svmsDir + "/" + objClassName + ".xml.gz";

	cv::FileStorage fs( svmFilename, cv::FileStorage::READ);
	if( fs.isOpened() )
	{
		std::cout << "*** LOADING SVM CLASSIFIER FOR CLASS " << objClassName << " ***" << std::endl;
		svm.load( svmFilename.c_str() );
	}
	else
	{
		std::cout << "*** TRAINING CLASSIFIER FOR CLASS " << objClassName << " ***" << std::endl;
		std::cout << "CALCULATING BOW VECTORS FOR TRAINING SET OF " << objClassName << "..." << std::endl;

		// Get classification ground truth for images in the training set
		std::vector<ObdImage> images;
		std::vector<cv::Mat> bowImageDescriptors;
		std::vector<char> objectPresent;
		vocData.getClassImages( objClassName, CV_OBD_TRAIN, images, objectPresent );

		// Compute the bag of words vector for each image in the training set.
		calculateImageDescriptors( images, bowImageDescriptors, bowExtractor, fdetector, resPath );

		// Remove any images for which descriptors could not be calculated
		removeEmptyBowImageDescriptors( images, bowImageDescriptors, objectPresent );

		CV_Assert( svmParamsExt.descPercent > 0.f && svmParamsExt.descPercent <= 1.f );
		if( svmParamsExt.descPercent < 1.f )
		{
			int descsToDelete = static_cast<int>(static_cast<float>(images.size())*(1.0-svmParamsExt.descPercent));

			std::cout << "Using " << (images.size() - descsToDelete) << " of " << images.size() <<
				" descriptors for training (" << svmParamsExt.descPercent*100.0 << " %)" << std::endl;
			removeBowImageDescriptorsByCount( images, bowImageDescriptors, objectPresent, svmParamsExt, descsToDelete );
		}

		// Prepare the input matrices for SVM training.
		cv::Mat trainData( images.size(), bowExtractor->getVocabulary().rows, CV_32FC1 );
		cv::Mat responses( images.size(), 1, CV_32SC1 );

		// Transfer bag of words vectors and responses across to the training data matrices
		for( size_t imageIdx = 0; imageIdx < images.size(); imageIdx++ )
		{
			// Transfer image descriptor (bag of words vector) to training data matrix
			cv::Mat submat = trainData.row(imageIdx);
			if( bowImageDescriptors[imageIdx].cols != bowExtractor->descriptorSize() )
			{
				std::cout << "Error: computed bow image descriptor size " << bowImageDescriptors[imageIdx].cols
					<< " differs from vocabulary size" << bowExtractor->getVocabulary().cols << std::endl;
				exit(-1);
			}
			bowImageDescriptors[imageIdx].copyTo( submat );

			// Set response value
			responses.at<int>(imageIdx) = objectPresent[imageIdx] ? 1 : -1;
		}

		std::cout << "TRAINING SVM FOR CLASS ..." << objClassName << "..." << std::endl;
		CvSVMParams svmParams;
		CvMat class_wts_cv;
		setSVMParams( svmParams, class_wts_cv, responses, svmParamsExt.balanceClasses );
		CvParamGrid c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid;
		setSVMTrainAutoParams( c_grid, gamma_grid,  p_grid, nu_grid, coef_grid, degree_grid );
		svm.train_auto( trainData, responses, cv::Mat(), cv::Mat(), svmParams, 10, c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid );
		std::cout << "SVM TRAINING FOR CLASS " << objClassName << " COMPLETED" << std::endl;

		svm.save( svmFilename.c_str() );
		std::cout << "SAVED CLASSIFIER TO FILE" << std::endl;
	}
}

void computeConfidences( CvSVM& svm, const std::string& objClassName, PascalVocDataset& vocData,
	cv::Ptr<cv::BOWImgDescriptorExtractor>& bowExtractor, const cv::Ptr<cv::FeatureDetector>& fdetector,
	const std::string& resPath )
{
	std::cout << "*** CALCULATING CONFIDENCES FOR CLASS " << objClassName << " ***" << std::endl;
	std::cout << "CALCULATING BOW VECTORS FOR TEST SET OF " << objClassName << "..." << std::endl;
	// Get classification ground truth for images in the test set
	std::vector<ObdImage> images;
	std::vector<cv::Mat> bowImageDescriptors;
	std::vector<char> objectPresent;
	vocData.getClassImages( objClassName, CV_OBD_TEST, images, objectPresent );

	// Compute the bag of words vector for each image in the test set
	calculateImageDescriptors( images, bowImageDescriptors, bowExtractor, fdetector, resPath );
	// Remove any images for which descriptors could not be calculated
	removeEmptyBowImageDescriptors( images, bowImageDescriptors, objectPresent);

	// Use the bag of words vectors to calculate classifier output for each image in test set
	std::cout << "CALCULATING CONFIDENCE SCORES FOR CLASS " << objClassName << "..." << std::endl;
	std::vector<float> confidences( images.size() );
	float signMul = 1.f;
	for( size_t imageIdx = 0; imageIdx < images.size(); imageIdx++ )
	{
		if( imageIdx == 0 )
		{
			// In the first iteration, determine the sign of the positive class
			float classVal = confidences[imageIdx] = svm.predict( bowImageDescriptors[imageIdx], false );
			float scoreVal = confidences[imageIdx] = svm.predict( bowImageDescriptors[imageIdx], true );
			signMul = (classVal < 0) == (scoreVal < 0) ? 1.f : -1.f;
		}
		// svm output of decision function
		confidences[imageIdx] = signMul * svm.predict( bowImageDescriptors[imageIdx], true );
	}

	std::cout << "WRITING QUERY RESULTS TO VOC RESULTS FILE FOR CLASS " << objClassName << "..." << std::endl;
	vocData.writeClassifierResultsFile( resPath + plotsDir, objClassName, CV_OBD_TEST, images, confidences, 1, true );

	std::cout << "DONE - " << objClassName << std::endl;
	std::cout << "---------------------------------------------------------------" << std::endl;
}

void computeGnuPlotOutput( const std::string& resPath, const std::string& objClassName, PascalVocDataset& vocData )
{
	std::vector<float> precision, recall;
	float ap;

	const std::string resultFile = vocData.getResultsFilename( objClassName, CV_VOC_TASK_CLASSIFICATION, CV_OBD_TEST);
	const std::string plotFile = resultFile.substr(0, resultFile.size()-4) + ".plt";

	std::cout << "Calculating precision recall curve for class '" <<objClassName << "'" << std::endl;
	vocData.calcClassifierPrecRecall( resPath + plotsDir + "/" + resultFile, precision, recall, ap, true );
	std::cout << "Outputting to GNUPlot file..." << std::endl;
	vocData.savePrecRecallToGnuplot( resPath + plotsDir + "/" + plotFile, precision, recall, ap, objClassName, CV_VOC_PLOT_PNG );
}

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

void bag_of_words()
{
	// Path to Pascal VOC data (e.g. /home/my/VOCdevkit/VOC2010). Note: VOC2007-VOC2010 are supported.
	const std::string vocPath("F:/archive_dataset/object_categorization/PASCAL_VOC/VOCtrainval_03-May-2010/VOCdevkit/VOC2010");
	// Path to result diractory. Following folders will be created in [result directory]: \n"
    //	bowImageDescriptors - to store image descriptors.
    //	svms - to store trained svms.
    //	plots - to store files for plots creating.
	const std::string resPath("./machine_vision_data/opencv/bow");

	// Read or set default parameters
	std::string vocName;
	local::DDMParams ddmParams;
	local::VocabTrainParams vocabTrainParams;
	//local::VocabTrainParams vocabTrainParams("chair", 1000, 200, 0.3);
	local::SVMTrainParamsExt svmTrainParamsExt;

	local::makeUsedDirs(resPath);

	cv::FileStorage paramsFS(resPath + "/" + local::paramsFile, cv::FileStorage::READ);
	if (paramsFS.isOpened())
	{
		local::readUsedParams(paramsFS.root(), vocName, ddmParams, vocabTrainParams, svmTrainParamsExt);
		CV_Assert(vocName == PascalVocDataset::getVocName(vocPath));
	}
	else
	{
		vocName = PascalVocDataset::getVocName(vocPath);

		// Feature detector name (e.g. SURF, FAST...)
		//	Currently 12/2010, this is FAST, STAR, SIFT, SURF, MSER, GFTT, HARRIS
		const std::string featDetectorName("SIFT");
		// Descriptor extractor name (e.g. SURF, SIFT)
		//	Currently 12/2010, this is SURF, OpponentSIFT, SIFT, OpponentSURF, BRIEF
		const std::string descExtracterName("SIFT");
		// Descriptor matcher name (e.g. BruteForce)
		//	Currently 12/2010, this is BruteForce, BruteForce-L1, FlannBased, BruteForce-Hamming, BruteForce-HammingLUT
		const std::string descMatcherName("FlannBased");

		ddmParams = local::DDMParams(featDetectorName, descExtracterName, descMatcherName);
		// vocabTrainParams and svmTrainParamsExt is set by defaults
		paramsFS.open(resPath + "/" + local::paramsFile, cv::FileStorage::WRITE);
		if (paramsFS.isOpened())
		{
			local::writeUsedParams(paramsFS, vocName, ddmParams, vocabTrainParams, svmTrainParamsExt);
			paramsFS.release();
		}
		else
		{
			std::cout << "File " << (resPath + "/" + local::paramsFile) << " can not be opened to write" << std::endl;
			return;
		}
	}

	// Create detector, descriptor, matcher.
	cv::Ptr<cv::FeatureDetector> featureDetector = cv::FeatureDetector::create(ddmParams.detectorType);
	cv::Ptr<cv::DescriptorExtractor> descExtractor = cv::DescriptorExtractor::create(ddmParams.descriptorType);
	cv::Ptr<cv::BOWImgDescriptorExtractor> bowExtractor;
	if (featureDetector.empty() || descExtractor.empty())
	{
		std::cout << "featureDetector or descExtractor was not created" << std::endl;
		return;
	}
	{
		cv::Ptr<cv::DescriptorMatcher> descMatcher = cv::DescriptorMatcher::create(ddmParams.matcherType);
		if (featureDetector.empty() || descExtractor.empty() || descMatcher.empty())
		{
			std::cout << "descMatcher was not created" << std::endl;
			return;
		}
		bowExtractor = new cv::BOWImgDescriptorExtractor(descExtractor, descMatcher);
	}

	// Print configuration to screen
	local::printUsedParams(vocPath, resPath, ddmParams, vocabTrainParams, svmTrainParamsExt);

	// Create object to work with VOC
	PascalVocDataset vocData(vocPath, false);

	// 1. Train visual word vocabulary if a pre-calculated vocabulary file doesn't already exist from previous run
	cv::Mat vocabulary = local::trainVocabulary(resPath + "/" + local::vocabularyFile, vocData, vocabTrainParams, featureDetector, descExtractor);
	bowExtractor->setVocabulary(vocabulary);

	// 2. Train a classifier and run a sample query for each object class
	const std::vector<std::string> &objClasses = vocData.getObjectClasses();  // object class list
	for (size_t classIdx = 0; classIdx < objClasses.size(); ++classIdx)
	{
		// Train a classifier on train dataset
		CvSVM svm;
		local::trainSVMClassifier(svm, svmTrainParamsExt, objClasses[classIdx], vocData, bowExtractor, featureDetector, resPath);

		// Now use the classifier over all images on the test dataset and rank according to score order also calculating precision-recall etc.
		local::computeConfidences(svm, objClasses[classIdx], vocData, bowExtractor, featureDetector, resPath);

		// Calculate precision/recall/ap and use GNUPlot to output to a pdf file
		local::computeGnuPlotOutput(resPath, objClasses[classIdx], vocData);
	}
}

}  // namespace my_opencv
