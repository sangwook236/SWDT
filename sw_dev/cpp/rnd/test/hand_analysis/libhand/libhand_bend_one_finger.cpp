// This is our little library for showing file dialogs
#include <libhand/file_dialog.h>
// We need the HandPose data structure
#include <libhand/hand_pose.h>
// ..the HandRenderer class which is used to render a hand
#include <libhand/hand_renderer.h>
// ..and SceneSpec which tells us where the hand 3D scene data is located on disk,
// and how the hand 3D object relates to our model of joints.
# include <libhand/scene_spec.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_libhand {

// [ref] ${LIBHAND_HOME}/examples/bend_one_finger.cpp
void bend_one_finger_example()
{
    // Setup the hand renderer
    libhand::HandRenderer hand_renderer;
    hand_renderer.Setup();

    // Ask the user to show the location of the scene spec file
    libhand::FileDialog dialog;
    dialog.SetTitle("Please select a scene spec file");
    std::string file_name = dialog.Open();

    // Process the scene spec file
    libhand::SceneSpec scene_spec(file_name);

    // Tell the renderer to load the scene
    hand_renderer.LoadScene(scene_spec);

    // Now we render a hand using a default pose
    hand_renderer.RenderHand();

    // Open a window through OpenCV
    std::string win_name("Hand Pic");
    cv::namedWindow(win_name);

    // We can get an OpenCV matrix from the rendered hand image
    cv::Mat pic = hand_renderer.pixel_buffer_cv();

    // And tell OpenCV to show the rendered hand
    cv::imshow(win_name, pic);
    cv::waitKey();

    // Now we're going to change the hand pose and render again
    // The hand pose depends on the number of bones, which is specified
    // by the scene spec file.
    libhand::FullHandPose hand_pose(scene_spec.num_bones());

    // We will bend the first joint, joint 0, by PI/2 radians (90 degrees)
    hand_pose.bend(0) += 3.14159 / 2;
    hand_renderer.SetHandPose(hand_pose);

    // Then we will render the hand again and show it to the user.
    hand_renderer.RenderHand();
    cv::imshow(win_name, pic);
    cv::waitKey();
}

}  // namespace my_libhand
