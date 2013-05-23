#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "IBGS.h"

class WeightedMovingVarianceBGS : public IBGS
{
private:
  bool firstTime;
  cv::Mat img_input_prev_1;
  cv::Mat img_input_prev_2;
  bool enableWeight;
  bool enableThreshold;
  int threshold;
  bool showOutput;

public:
  WeightedMovingVarianceBGS();
  ~WeightedMovingVarianceBGS();

  cv::Mat computeWeightedMean(const std::vector<cv::Mat> &v_img_input_f, const std::vector<double> weights);
  cv::Mat computeWeightedVariance(const cv::Mat &img_input_f, const cv::Mat &img_mean_f, const double weight);

  void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);

private:
  void saveConfig();
  void loadConfig();
};
