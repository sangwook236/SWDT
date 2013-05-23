#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "BGModelFuzzyGauss.h"

#include "../IBGS.h"

using namespace lb_library;
using namespace lb_library::FuzzyGaussian;

class LBFuzzyGaussian : public IBGS
{
private:
  bool firstTime;
  bool showOutput;
  
  BGModel* m_pBGModel;
  int sensitivity;
  int bgThreshold;
  int learningRate;
  int noiseVariance;

  cv::Mat img_foreground;
  cv::Mat img_background;

public:
  LBFuzzyGaussian();
  ~LBFuzzyGaussian();

  void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);
  //void finish(void);

private:
  void saveConfig();
  void loadConfig();
};