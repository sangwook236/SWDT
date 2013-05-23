#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "../IBGS.h"
#include "GrimsonGMM.h"

using namespace Algorithms::BackgroundSubtraction;

class DPGrimsonGMMBGS : public IBGS
{
private:
  bool firstTime;
  long frameNumber;
  IplImage* frame;
  RgbImage frame_data;

  GrimsonParams params;
  GrimsonGMM bgs;
  BwImage lowThresholdMask;
  BwImage highThresholdMask;

  double threshold;
  double alpha;
  int gaussians;
  bool showOutput;

public:
  DPGrimsonGMMBGS();
  ~DPGrimsonGMMBGS();

  void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);

private:
  void saveConfig();
  void loadConfig();
};

