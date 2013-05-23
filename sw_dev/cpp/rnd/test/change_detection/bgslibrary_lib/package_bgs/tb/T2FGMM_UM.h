#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "../IBGS.h"
#include "T2FGMM.h"

using namespace Algorithms::BackgroundSubtraction;

class T2FGMM_UM : public IBGS
{
private:
  bool firstTime;
  long frameNumber;
  IplImage* frame;
  RgbImage frame_data;

  T2FGMMParams params;
  T2FGMM bgs;
  BwImage lowThresholdMask;
  BwImage highThresholdMask;

  double threshold;
  double alpha;
  float km;
  float kv;
  int gaussians;
  bool showOutput;

public:
  T2FGMM_UM();
  ~T2FGMM_UM();

  void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);

private:
  void saveConfig();
  void loadConfig();
};

