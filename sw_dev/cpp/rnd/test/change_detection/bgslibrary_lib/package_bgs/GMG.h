#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

#include "IBGS.h"

class GMG : public IBGS
{
private:
  bool firstTime;
  cv::Ptr<cv::BackgroundSubtractorGMG> fgbg;
  int initializationFrames;
  double decisionThreshold;
  cv::Mat img_foreground;
  cv::Mat img_segmentation;
  bool showOutput;

public:
  GMG();
  ~GMG();

  void process(const cv::Mat &img_input, cv::Mat &img_output);

private:
  void saveConfig();
  void loadConfig();
};

