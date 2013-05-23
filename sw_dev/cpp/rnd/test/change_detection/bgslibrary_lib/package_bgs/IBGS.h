#pragma once

#include <cv.h>

class IBGS
{
public:
  virtual void process(const cv::Mat &img_input, cv::Mat &img_foreground, cv::Mat &img_background) = 0;
  /*virtual void process(const cv::Mat &img_input, cv::Mat &img_foreground){
    process(img_input, img_foreground, cv::Mat());
  }*/
  virtual ~IBGS(){}

private:
  virtual void saveConfig() = 0;
  virtual void loadConfig() = 0;
};
