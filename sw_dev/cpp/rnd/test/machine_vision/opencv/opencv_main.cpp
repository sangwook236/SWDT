//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/core/core.hpp>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace opencv {

void print_opencv_matrix(const CvMat *mat)
{
	if ((mat->type & CV_32F) == CV_32F || (mat->type & CV_64F) == CV_64F)
	{
		for (int i = 0; i < mat->height; ++i)
		{
			for (int j = 0; j < mat->width; ++j)
				std::wcout << cvmGet(mat, i, j) << L" ";
			std::wcout << std::endl;
		}
	}
	else if ((mat->type & CV_8U) == CV_8U || (mat->type & CV_8S) == CV_8S ||
		(mat->type & CV_16U) == CV_16U || (mat->type & CV_16S) == CV_16S ||
		(mat->type & CV_32S) == CV_32S)
	{
		for (int i = 0; i < mat->height; ++i)
		{
			for (int j = 0; j < mat->width; ++j)
				//std::wcout << cvGet2D(mat, i, j).val[0] << L" ";
				std::wcout << CV_MAT_ELEM(*mat, int, i, j) << L" ";
			std::wcout << std::endl;
		}
	}
}

void print_opencv_matrix(const cv::Mat &mat)
{
	std::cout << mat << std::endl;
	//std::cout << cv::format(mat, "python") << std::endl;
	//std::cout << cv::format(mat, "numpy") << std::endl;
	//std::cout << cv::format(mat, "csv") << std::endl;
	//std::cout << cv::format(mat, "C") << std::endl;
}

void text_output();
void matrix_operation();
void matrix_operation_using_gpu();
void vector_operation();
void image_operation();
void image_sequence();
void image_conversion();
void image_filtering();
void image_filtering_using_gpu();
void image_processing_using_gpu();
void color_filtering();
void color_correction();
void skin_color_filtering();
void histogram();
void histogram_using_gpu();
void convolution_correlation();
void fourier_transform();
void morphological_operation();
void image_pyramid();
void image_gradient();
void edge_detection();
void distance_transform();
void convex_hull();
void hough_transform();
void template_matching();
void chamfer_matching();
void shape_finding();
void shape_matching();
void snake();
void segmentation();
void outlier_removal();
void feature_extraction();
void feature_description();
void feature_matching();
void feature_extraction_and_matching();
void feature_extraction_and_matching_by_signature();
void feature_extraction_and_matching_using_gpu();
void generic_description_and_matching();
void bag_of_words();
void pca();
void clustering();
void train_by_svm();
void train_by_ann();
void object_detection();
void face_detection();
void face_detection_using_gpu();
void human_detection();
void human_detection_using_gpu();
void camera_geometry();
void homography();
void image_labeling_using_gpu();
void stereo_matching();
void stereo_matching_using_gpu();
void change_detection();
void object_tracking();
void kalman_filtering();
void optical_flow();
void motion_history_image();

void iterative_closest_point();

void hand_pose_estimation();
void hand_detection();
void motion_segmentation();
void gesture_recognition();

}  // namespace opencv

int opencv_main(int argc, char *argv[])
{
	try
	{
		//opencv::text_output();

		//opencv::matrix_operation();
		//opencv::matrix_operation_using_gpu();  // not yet implemented
		//opencv::vector_operation();
		//opencv::image_operation();
		//opencv::image_conversion();
		//opencv::image_sequence();

		//opencv::image_filtering();
		//opencv::image_filtering_using_gpu();  // not yet implemented
		//opencv::image_processing_using_gpu();  // not yet implemented
		//opencv::color_filtering();
		//opencv::color_correction();
		//opencv::skin_color_filtering();
		//opencv::histogram();
		//opencv::histogram()_using_gpu();  // not yet implemented

		//opencv::convolution_correlation();
		//opencv::fourier_transform();
		//opencv::morphological_operation();
		//opencv::image_pyramid();

		//opencv::image_gradient();
		//opencv::edge_detection();

		//opencv::distance_transform();
		//opencv::convex_hull();
		//opencv::hough_transform();

		//opencv::template_matching();
		//opencv::chamfer_matching();
		//opencv::shape_finding();
		//opencv::shape_matching();

		//opencv::snake();
		//opencv::segmentation();

		//opencv::outlier_removal();

		//opencv::feature_extraction();
		//opencv::feature_description();
		//opencv::feature_matching();
		//opencv::feature_extraction_and_matching();
		//opencv::feature_extraction_and_matching_by_signature();
		//opencv::feature_extraction_and_matching_using_gpu();  // not yet implemented
		//opencv::generic_description_and_matching();

		//opencv::bag_of_words();

		//opencv::pca();

		//opencv::clustering();
		//opencv::train_by_svm();
		//opencv::train_by_ann();

		//opencv::object_detection();  // not yet implemented
		//opencv::face_detection();
		//opencv::face_detection_using_gpu();  // not yet implemented
		//opencv::human_detection();
		//opencv::human_detection_using_gpu();  // not yet implemented

		//opencv::camera_geometry();
		//opencv::homography();

		//opencv::image_labeling_using_gpu();  // not yet implemented
		//opencv::stereo_matching();
		//opencv::stereo_matching_using_gpu();  // not yet implemented

		//opencv::change_detection();

		//opencv::object_tracking();
		//opencv::kalman_filtering();

		//opencv::optical_flow();
		//opencv::motion_history_image();

		//----------------------------------------------
		// extension

		//opencv::iterative_closest_point();

		//----------------------------------------------
		// application

		//opencv::hand_pose_estimation();
		//opencv::hand_detection();

		//opencv::motion_segmentation();
		opencv::gesture_recognition();
	}
	catch (const cv::Exception &e)
	{
		//std::cout << "OpenCV exception occurred !!!: " << e.what() << std::endl;
		//std::cout << "OpenCV exception occurred !!!: " << cvErrorStr(e.code) << std::endl;
		std::cout << "OpenCV exception occurred !!!:" << std::endl
			<< "\tdescription: " << e.err << std::endl
			<< "\tline:        " << e.line << std::endl
			<< "\tfunction:    " << e.func << std::endl
			<< "\tfile:        " << e.file << std::endl;

		return -1;
	}

    return 0;
}
