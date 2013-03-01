#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "../IBGS.h"
#include "Eigenbackground.h"

using namespace Algorithms::BackgroundSubtraction;

class DPEigenbackgroundBGS : public IBGS
{
private:
  bool firstTime;
  long frameNumber;
  IplImage* frame;
  RgbImage frame_data;

  EigenbackgroundParams params;
  Eigenbackground bgs;
  BwImage lowThresholdMask;
  BwImage highThresholdMask;

  int threshold;
  int historySize;
  int embeddedDim;
  bool showOutput;

public:
  DPEigenbackgroundBGS();
  ~DPEigenbackgroundBGS();

  void process(const cv::Mat &img_input, cv::Mat &img_output);

private:
  void saveConfig();
  void loadConfig();
};

