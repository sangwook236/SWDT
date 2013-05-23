#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "../IBGS.h"
#include "MRF.h"

using namespace Algorithms::BackgroundSubtraction;

class T2FMRF_UV : public IBGS
{
private:
  bool firstTime;
  long frameNumber;
  IplImage *frame;
  RgbImage frame_data;

  IplImage *old_labeling;
  IplImage *old;

  T2FMRFParams params;
  T2FMRF bgs;
  BwImage lowThresholdMask;
  BwImage highThresholdMask;

  double threshold;
  double alpha;
  float km;
  float kv;
  int gaussians;
  bool showOutput;

  MRF_TC mrf;
  GMM *gmm;
  HMM *hmm;

public:
  T2FMRF_UV();
  ~T2FMRF_UV();

  void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);

private:
  void saveConfig();
  void loadConfig();
};
