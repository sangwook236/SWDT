#pragma once

#include <iostream>
#include <cv.h>
#include <highgui.h>

#include "../IBGS.h"
#include "TextureBGS.h"
//#include "ConnectedComponents.h"

class DPTextureBGS : public IBGS
{
private:
  bool firstTime;
  bool showOutput;

  int width;
  int height;
  int size;
  TextureBGS bgs;
  IplImage* frame;
  RgbImage image;
  BwImage fgMask;
  BwImage tempMask;
  TextureArray* bgModel;
  RgbImage texture;
  unsigned char* modeArray;
  TextureHistogram* curTextureHist;
  //ConnectedComponents cc;
  //CBlobResult largeBlobs;
  //IplConvKernel* dilateElement;
  //IplConvKernel* erodeElement;
  //bool enableFiltering;

public:
  DPTextureBGS();
  ~DPTextureBGS();

  void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);

private:
  void saveConfig();
  void loadConfig();
};
