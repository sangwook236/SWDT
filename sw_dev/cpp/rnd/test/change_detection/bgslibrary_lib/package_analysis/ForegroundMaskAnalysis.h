#pragma once

#include <iostream>
#include <string>
#include <cv.h>
#include <highgui.h>

class ForegroundMaskAnalysis
{
private:
  bool firstTime;
  bool showOutput;

public:
  ForegroundMaskAnalysis();
  ~ForegroundMaskAnalysis();
  
  long stopAt;
  std::string img_ref_path;

  void process(const long &frameNumber, const std::string &name, const cv::Mat &img_input);

private:
  void saveConfig();
  void loadConfig();
};