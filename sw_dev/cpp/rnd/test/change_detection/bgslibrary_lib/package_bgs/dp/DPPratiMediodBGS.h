#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "../IBGS.h"
#include "PratiMediodBGS.h"

using namespace Algorithms::BackgroundSubtraction;

class DPPratiMediodBGS : public IBGS
{
private:
  bool firstTime;
  long frameNumber;
  IplImage* frame;
  RgbImage frame_data;

  PratiParams params;
  PratiMediodBGS bgs;
  BwImage lowThresholdMask;
  BwImage highThresholdMask;

  int threshold;
  int samplingRate;
  int historySize;
  int weight;
  bool showOutput;

public:
  DPPratiMediodBGS();
  ~DPPratiMediodBGS();

  void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);

private:
  void saveConfig();
  void loadConfig();
};

