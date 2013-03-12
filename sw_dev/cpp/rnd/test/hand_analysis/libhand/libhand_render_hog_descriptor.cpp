// See bend_one_finger.cc for an explanation of what these are
#include <libhand/file_dialog.h>
#include <libhand/hand_pose.h>
#include <libhand/hand_renderer.h>
#include <libhand/scene_spec.h>
// We need the HogDescriptor class to store the HoG data
#include <libhand/hog_descriptor.h>
// We also use HogUtils to provide for a convenient visualization of HoG data
#include <libhand/hog_utils.h>
// The ImageToHogCalculator calculates a HoG descriptor from an input image
#include <libhand/image_to_hog_calculator.h>
// The ImageUtils class contains functions that make it easier to manipulate OpenCV matrices as images with masks.
// It takes care of issues such as grayscale vs. RGB, contours, etc.
#include <libhand/image_utils.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_libhand {

// [ref] ${LIBHAND_HOME}/examples/render_hog_descriptor.cpp
void render_hog_descriptor_example()
{
    // See bend_one_finger.cc for an explanation of what this part of code does
    libhand::HandRenderer hand_renderer;
    hand_renderer.Setup();

    libhand::FileDialog dialog;
    dialog.SetTitle("Please select a scene spec file");

    libhand::SceneSpec scene_spec(dialog.Open());
    hand_renderer.LoadScene(scene_spec);
    hand_renderer.RenderHand();

    // We use the rendered hand as an input image to the HoG calculator
    cv::Mat in_image = hand_renderer.pixel_buffer_cv();

    // This utility routine flags all the non-black pixels as relevant
    // for the HoG calculation. The Region of Interest (ROI) will be
    // calculated from this mask.
    cv::Mat image_mask = libhand::ImageUtils::MaskFromNonZero(in_image);

    // Get the bounding box for the ROI. The bounding box will be around
    // the non-zero pixels in the image.
    cv::Rect roi_box = libhand::ImageUtils::FindBoundingBox(image_mask);

    // Make a reference roi_image to the part of the image bounded by roi_box.
    // The coordinate 0,0 in roi_image will correspond to the top-left corner
    // of the roi_box bounding box.
    cv::Mat roi_image(in_image, roi_box);

    // Similarly make a reference to the image_mask.
    cv::Mat roi_mask(image_mask, roi_box);

    // Our output image will contain the input image as a background.
    cv::Mat out_image = in_image.clone();

    // We declare a HoG descriptor:
    // - it's an array of cells with 5 rows and 7 columns
    // - each cell contains a 12-binned histogram of gradients
    libhand::HogDescriptor hog_desc(5, 7, 12);

    // We calculate the hog_desc descriptor from the interesting part of
    // the image -- the ROI.
    libhand::ImageToHogCalculator hog_calc;
    hog_calc.CalcHog(roi_image, roi_mask, &hog_desc);

    // We call a utility routine to visualise the hog_descriptor hog_desc
    // by drawing on top of the output image out_image
    libhand::HogUtils::RenderHog(roi_image, hog_desc, out_image);

    // We show the result to the user and wait for a key
    const std::string win_name("Hand HoG");
    cv::namedWindow(win_name);
    cv::imshow(win_name, out_image);

    cv::waitKey(0);
}

}  // namespace my_libhand
