#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "../IBGS.h"
#include "AdaptiveMedianBGS.h"

using namespace Algorithms::BackgroundSubtraction;

class DPAdaptiveMedianBGS : public IBGS
{
private:
  bool firstTime;
  long frameNumber;
  IplImage* frame;
  RgbImage frame_data;

  AdaptiveMedianParams params;
  AdaptiveMedianBGS bgs;
  BwImage lowThresholdMask;
  BwImage highThresholdMask;

  int threshold;
  int samplingRate;
  int learningFrames;
  bool showOutput;

public:
  DPAdaptiveMedianBGS();
  ~DPAdaptiveMedianBGS();

  void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);

private:
  void saveConfig();
  void loadConfig();
};

