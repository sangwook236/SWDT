#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "BGModelGauss.h"

#include "../IBGS.h"

using namespace lb_library;
using namespace lb_library::SimpleGaussian;

class LBSimpleGaussian : public IBGS
{
private:
  bool firstTime;
  bool showOutput;
  
  BGModel* m_pBGModel;
  int sensitivity;
  int noiseVariance;
  int learningRate;

  cv::Mat img_foreground;
  cv::Mat img_background;

public:
  LBSimpleGaussian();
  ~LBSimpleGaussian();

  void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);
  //void finish(void);

private:
  void saveConfig();
  void loadConfig();
};