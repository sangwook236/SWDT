#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <opencv2/video/background_segm.hpp>

#include "IBGS.h"

class MixtureOfGaussianV2BGS : public IBGS
{
private:
  bool firstTime;
  cv::BackgroundSubtractorMOG2 mog;
  cv::Mat img_foreground;
  double alpha;
  bool enableThreshold;
  int threshold;
  bool showOutput;

public:
  MixtureOfGaussianV2BGS();
  ~MixtureOfGaussianV2BGS();

  void process(const cv::Mat &img_input, cv::Mat &img_output);

private:
  void saveConfig();
  void loadConfig();
};

