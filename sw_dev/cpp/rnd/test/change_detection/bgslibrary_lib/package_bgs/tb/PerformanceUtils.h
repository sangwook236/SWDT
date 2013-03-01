#pragma once
/*
Code provided by Thierry BOUWMANS

Maitre de Conférences
Laboratoire MIA
Université de La Rochelle
17000 La Rochelle
France
tbouwman@univ-lr.fr

http://sites.google.com/site/thierrybouwmans/
*/
#include <stdio.h>
#include <fstream>
#include <cv.h>
#include <highgui.h>

#include "PixelUtils.h"

class PerformanceUtils
{
public:
  PerformanceUtils(void);
  ~PerformanceUtils(void);

  float NrPixels(IplImage *image);
  float NrAllDetectedPixNotNULL(IplImage *image, IplImage *ground_truth);
  float NrTruePositives(IplImage *image, IplImage *ground_truth, bool debug = false);
  float NrTrueNegatives(IplImage *image, IplImage *ground_truth, bool debug = false);
  float NrFalsePositives(IplImage *image, IplImage *ground_truth, bool debug = false);
  float NrFalseNegatives(IplImage *image, IplImage *ground_truth, bool debug = false);
  float SimilarityMeasure(IplImage *image, IplImage *ground_truth, bool debug = false);

  void ImageROC(IplImage *image, IplImage* ground_truth, bool saveResults = false, char* filename = "");
  void PerformanceEvaluation(IplImage *image, IplImage *ground_truth, bool saveResults = false, char* filename = "", bool debug = false);
};

