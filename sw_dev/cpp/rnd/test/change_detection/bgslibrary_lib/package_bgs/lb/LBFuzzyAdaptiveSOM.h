#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "BGModelFuzzySom.h"

#include "../IBGS.h"

using namespace lb_library;
using namespace lb_library::FuzzyAdaptiveSOM;

class LBFuzzyAdaptiveSOM : public IBGS
{
private:
  bool firstTime;
  bool showOutput;
  
  BGModel* m_pBGModel;
  int sensitivity;
  int trainingSensitivity;
  int learningRate;
  int trainingLearningRate;
  int trainingSteps;

  cv::Mat img_foreground;
  cv::Mat img_background;

public:
  LBFuzzyAdaptiveSOM();
  ~LBFuzzyAdaptiveSOM();

  void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);
  //void finish(void);

private:
  void saveConfig();
  void loadConfig();
};