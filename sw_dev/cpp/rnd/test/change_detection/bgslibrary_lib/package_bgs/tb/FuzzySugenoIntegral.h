#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "../IBGS.h"

#include "FuzzyUtils.h"

class FuzzySugenoIntegral : public IBGS
{
private:
  bool firstTime;
  long long frameNumber;
  bool showOutput;
  
  int framesToLearn;
  double alphaLearn;
  double alphaUpdate;
  int colorSpace;
  int option;
  bool smooth;
  double threshold;

  FuzzyUtils fu;
  cv::Mat img_background_f3;
  
public:
  FuzzySugenoIntegral();
  ~FuzzySugenoIntegral();

  void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);

private:
  void saveConfig();
  void loadConfig();
};

