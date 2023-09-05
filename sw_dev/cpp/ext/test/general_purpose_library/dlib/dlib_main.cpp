//#include "stdafx.h"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <string>
#include <vector>


namespace {
namespace local {

// REF [site] >> http://dlib.net/face_alignment.py.html
// REF [file] >> ${DLIB_HOME}/examples/face_landmark_detection_ex.cpp
void face_alignment_example()
{
	// REF [site] >> https://github.com/davisking/dlib-models
	const std::string predictor_filepath("./shape_predictor_5_face_landmarks.dat");
	//const std::string predictor_filepath("./shape_predictor_68_face_landmarks.dat");
	// REF [directory] >> ${DLIB_HOME}/examples/faces
	const std::string face_file_filepath("./bald_guys.jpg");
	//const std::string face_file_filepath("./Tom_Cruise_avp_2014_4.jpg");

	// Load all the models we need: a detector to find the faces, a shape predictor to find face landmarks so we can precisely localize the face.
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor sp;
	dlib::deserialize(predictor_filepath) >> sp;

	// Load the image using Dlib.
	dlib::array2d<dlib::rgb_pixel> img;
	dlib::load_image(img, face_file_filepath);

	// Ask the detector to find the bounding boxes of each face.
	// The 1 in the second argument indicates that we should upsample the image 1 time.
	// This will make everything bigger and allow us to detect more faces.
	const std::vector<dlib::rectangle> dets = detector(img, 1);

	const size_t num_faces = dets.size();
	if (dets.empty())
	{
		std::cerr << "No faces found in " << face_file_filepath << std::endl;
		return;
	}

	// Find the 5 face landmarks we need to do the alignment.
	std::vector<dlib::full_object_detection> faces;
	faces.reserve(num_faces);
	for (const auto &detection : dets)
		faces.push_back(sp(img, detection));

	dlib::image_window window;

	// Get the aligned face images.
	dlib::array<dlib::array2d<dlib::rgb_pixel>> face_chips;
	//dlib::extract_image_chips(img, dlib::get_face_chip_details(faces, 160, 0.25), face_chips);
	dlib::extract_image_chips(img, dlib::get_face_chip_details(faces, 320), face_chips);
	window.set_image(tile_images(face_chips));

	std::cin.get();

	// It is also possible to get a single chip.
	dlib::array2d<dlib::rgb_pixel> face_chip;
	dlib::extract_image_chip(img, dlib::get_face_chip_details(faces[0], 320), face_chip);
	window.set_image(face_chip);

	std::cin.get();
}

}  // namespace local
}  // unnamed namespace

namespace my_dlib {

void optimization_example();

void max_cost_assignment_example();
void graph_labeling_example();

void svm_struct_example();
void model_selection_example();
void dnn_example();

}  // namespace my_dlib

int dlib_main(int argc, char *argv[])
{
	// Matrix operation.
	// REF [file] >> ${DLIB_HOME}/examples/matrix_ex.cpp

	// Optimization.
	//my_dlib::optimization_example();

	// Assignment problem (use Hungarian/Kuhn-Munkres algorithm).
	//my_dlib::max_cost_assignment_example();

	// Graph labeling.
	//my_dlib::graph_labeling_example();

	// Structured SVM.
	//my_dlib::svm_struct_example();

	// Model selection.
	//my_dlib::model_selection_example();

	// Deep learning.
	//my_dlib::dnn_example();

	// Face alignment.
	local::face_alignment_example();

	// Control.
	//my_dlib::control_example();  // Not yet tested.

	return 0;
}
