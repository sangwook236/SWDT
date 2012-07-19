//#include "stdafx.h"
#if defined(WIN32)
#include <vld/vld.h>
#endif
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/core/core.hpp>
#include <iostream>
#include <ctime>


void print_opencv_matrix(const CvMat *mat);
void print_opencv_matrix(const cv::Mat &mat);

int main(int argc, char **argv)
{
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

	try
	{
		std::srand((unsigned int)std::time(NULL));

		//text_output();

		//matrix_operation();
		//matrix_operation_using_gpu();  // not yet implemented
		//vector_operation();
		//image_operation();
		//image_conversion();
		//image_sequence();

		//image_filtering();
		//image_filtering_using_gpu();  // not yet implemented
		//image_processing_using_gpu();  // not yet implemented
		//color_filtering();
		//color_correction();
		//skin_color_filtering();
		//histogram();
		//histogram()_using_gpu();  // not yet implemented

		//convolution_correlation();
		//fourier_transform();
		//morphological_operation();
		//image_pyramid();

		//image_gradient();
		//edge_detection();

		//distance_transform();
		//convex_hull();
		//hough_transform();

		//template_matching();
		//chamfer_matching();
		//shape_finding();
		//shape_matching();

		//snake();
		//segmentation();

		//outlier_removal();

		//feature_extraction();
		//feature_description();
		//feature_matching();
		//feature_extraction_and_matching();
		//feature_extraction_and_matching_by_signature();
		//feature_extraction_and_matching_using_gpu();  // not yet implemented
		//generic_description_and_matching();

		//bag_of_words();

		//pca();

		//clustering();
		//train_by_svm();
		//train_by_ann();

		//object_detection();  // not yet implemented
		//face_detection();
		//face_detection_using_gpu();  // not yet implemented
		//human_detection();
		//human_detection_using_gpu();  // not yet implemented

		//camera_geometry();
		//homography();

		//image_labeling_using_gpu();  // not yet implemented
		//stereo_matching();
		//stereo_matching_using_gpu();  // not yet implemented

		//change_detection();

		//object_tracking();
		//kalman_filtering();

		//optical_flow();
		//motion_history_image();

		//----------------------------------------------
		// extension

		//iterative_closest_point();

		//----------------------------------------------
		// application

		//hand_pose_estimation();
		//hand_detection();

		//motion_segmentation();
		gesture_recognition();
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
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred !!!: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred !!!" << std::endl;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

    return 0;
}

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
