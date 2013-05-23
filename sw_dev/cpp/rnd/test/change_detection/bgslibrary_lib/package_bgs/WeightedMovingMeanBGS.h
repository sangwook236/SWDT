#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "IBGS.h"

class WeightedMovingMeanBGS : public IBGS
{
private:
  bool firstTime;
  cv::Mat img_input_prev_1;
  cv::Mat img_input_prev_2;
  bool enableWeight;
  bool enableThreshold;
  int threshold;
  bool showOutput;
  bool showBackground;

public:
  WeightedMovingMeanBGS();
  ~WeightedMovingMeanBGS();

  void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);

private:
  void saveConfig();
  void loadConfig();
};

