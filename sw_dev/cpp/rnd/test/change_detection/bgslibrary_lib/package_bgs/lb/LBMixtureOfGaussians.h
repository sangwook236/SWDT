#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "BGModelMog.h"

#include "../IBGS.h"

using namespace lb_library;
using namespace lb_library::MixtureOfGaussians;

class LBMixtureOfGaussians : public IBGS
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
  LBMixtureOfGaussians();
  ~LBMixtureOfGaussians();

  void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);
  //void finish(void);

private:
  void saveConfig();
  void loadConfig();
};