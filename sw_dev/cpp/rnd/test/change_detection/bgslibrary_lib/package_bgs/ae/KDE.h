#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "NPBGSubtractor.h"
#include "..\IBGS.h"

class KDE : public IBGS
{
private:
  NPBGSubtractor *p;
  int rows;
  int cols;
  int color_channels;
  int SequenceLength;
  int TimeWindowSize;
  int SDEstimationFlag;
  int lUseColorRatiosFlag;
  double th;
  double alpha;
  int framesToLearn;
  int frameNumber;
  bool firstTime;
  bool showOutput;

  cv::Mat img_foreground;
  unsigned char *FGImage;
  unsigned char *FilteredFGImage;
  unsigned char **DisplayBuffers;

public:
  KDE();
  ~KDE();

  void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);
  
private:
  void saveConfig();
  void loadConfig();
};
