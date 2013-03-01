#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "../IBGS.h"
#include "MeanBGS.h"

using namespace Algorithms::BackgroundSubtraction;

class DPMeanBGS : public IBGS
{
private:
  bool firstTime;
  long frameNumber;
  IplImage* frame;
  RgbImage frame_data;

  MeanParams params;
  MeanBGS bgs;
  BwImage lowThresholdMask;
  BwImage highThresholdMask;

  int threshold;
  double alpha;
  int learningFrames;
  bool showOutput;

public:
  DPMeanBGS();
  ~DPMeanBGS();

  void process(const cv::Mat &img_input, cv::Mat &img_output);

private:
  void saveConfig();
  void loadConfig();
};

