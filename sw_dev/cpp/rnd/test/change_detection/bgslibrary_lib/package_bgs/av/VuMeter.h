#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "TBackgroundVuMeter.h"
#include "../IBGS.h"

class VuMeter : public IBGS
{
private:
  TBackgroundVuMeter bgs;

  IplImage *frame;
  IplImage *gray;
  IplImage *background;
  IplImage *mask;
  
  bool firstTime;
  bool showOutput;
  bool enableFilter;
  
  int binSize;
  double alpha;
  double threshold;
  
public:
  VuMeter();
  ~VuMeter();

  void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);

private:
  void saveConfig();
  void loadConfig();
};
