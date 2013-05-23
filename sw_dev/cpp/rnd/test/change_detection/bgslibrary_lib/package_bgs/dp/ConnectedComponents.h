/****************************************************************************
*
* ConnectedComponents.h
*
* Purpose: 	Find connected components in an image. This class effectively just
*			encapsulates functionality found in cvBlobLib.
*
* Author: Donovan Parks, August 2007
*
******************************************************************************/

#ifndef _CONNECTED_COMPONENTS_H_
#define _CONNECTED_COMPONENTS_H_

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include "BlobResult.h"
#include "Image.h"

class ConnectedComponents
{
public:
  ConnectedComponents(){}
  ~ConnectedComponents(){}
  
  void SetImage(BwImage* image) { m_image = image; }

  void Find(int threshold);

  void FilterMinArea(int area, CBlobResult& largeBlobs);

  void GetBlobImage(RgbImage& blobImage);
  void SaveBlobImage(char* filename);

  void GetComponents(RgbImage& blobImage);

  void FilterSaliency(BwImage& highThreshold, RgbImage& blobImg, float minSaliency, 
                          CBlobResult& salientBlobs, CBlobResult& unsalientBlobs);

  void ColorBlobs(IplImage* image, CBlobResult& blobs, CvScalar& color);

private:
  BwImage* m_image;

  CBlobResult m_blobs;
};

#endif
