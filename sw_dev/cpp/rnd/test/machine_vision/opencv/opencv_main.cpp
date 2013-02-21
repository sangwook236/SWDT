//#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/core/core.hpp>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_opencv {

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
void change_detection_using_gpu();
void object_tracking();
void kalman_filtering();
void optical_flow();
void motion_history_image();

void iterative_closest_point();

void hand_detection();
void hand_pose_estimation();
void motion_segmentation();
void gesture_recognition();

}  // namespace my_opencv

int opencv_main(int argc, char *argv[])
{
	try
	{
		//opencv::text_output();

		//my_opencv::matrix_operation();
		//my_opencv::matrix_operation_using_gpu();  // not yet implemented
		//my_opencv::vector_operation();
		//my_opencv::image_operation();
		//my_opencv::image_conversion();
		//my_opencv::image_sequence();

		//my_opencv::image_filtering();
		//my_opencv::image_filtering_using_gpu();  // not yet implemented
		//my_opencv::image_processing_using_gpu();  // not yet implemented
		//my_opencv::color_filtering();
		//my_opencv::color_correction();
		//my_opencv::skin_color_filtering();
		//my_opencv::histogram();
		//my_opencv::histogram()_using_gpu();  // not yet implemented

		//my_opencv::convolution_correlation();
		//my_opencv::fourier_transform();
		//my_opencv::morphological_operation();
		//my_opencv::image_pyramid();

		//my_opencv::image_gradient();
		//my_opencv::edge_detection();

		//my_opencv::distance_transform();
		//my_opencv::convex_hull();
		//my_opencv::hough_transform();

		//my_opencv::template_matching();
		//my_opencv::chamfer_matching();
		//my_opencv::shape_finding();
		//my_opencv::shape_matching();

		//my_opencv::snake();
		//my_opencv::segmentation();

		//my_opencv::outlier_removal();

		//my_opencv::feature_extraction();
		//my_opencv::feature_description();
		//my_opencv::feature_matching();
		//my_opencv::feature_extraction_and_matching();
		//my_opencv::feature_extraction_and_matching_by_signature();
		//my_opencv::feature_extraction_and_matching_using_gpu();  // not yet implemented
		//my_opencv::generic_description_and_matching();

		//my_opencv::bag_of_words();

		//my_opencv::pca();

		//my_opencv::clustering();
		//my_opencv::train_by_svm();
		//my_opencv::train_by_ann();

		//my_opencv::object_detection();  // not yet implemented
		//my_opencv::face_detection();
		//my_opencv::face_detection_using_gpu();  // not yet implemented
		//my_opencv::human_detection();
		//my_opencv::human_detection_using_gpu();  // not yet implemented

		//my_opencv::camera_geometry();
		//my_opencv::homography();

		//my_opencv::image_labeling_using_gpu();  // not yet implemented
		//my_opencv::stereo_matching();
		//my_opencv::stereo_matching_using_gpu();  // not yet implemented

		//my_opencv::change_detection();
		//my_opencv::change_detection_using_gpu();  // not yet implemented

		//my_opencv::object_tracking();
		//my_opencv::kalman_filtering();

		//my_opencv::optical_flow();
		//my_opencv::optical_flow_using_gpu();  // not yet implemented
		//my_opencv::motion_history_image();

		//----------------------------------------------
		// extension

		//my_opencv::iterative_closest_point();

		//----------------------------------------------
		// application

		my_opencv::hand_detection();
		//my_opencv::hand_pose_estimation();

		//my_opencv::motion_segmentation();
		//my_opencv::gesture_recognition();
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

		return 1;
	}

    return 0;
}
