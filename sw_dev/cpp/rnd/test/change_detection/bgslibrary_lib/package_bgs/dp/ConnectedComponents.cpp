/****************************************************************************
*
* ConnectedComponents.cpp
*
* Purpose: Find connected components in an image. This class effectively just
*					 encapsulates functionality found in cvBlobLib.
*
* Author: Donovan Parks, August 2007
*
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include "ConnectedComponents.h"
#include "Error.h"

void ConnectedComponents::Find(int threshold)
{
  // set border of image to black so components along the border are 
  // not treated as a single component (this is a hack to fix a bug in cvBlobsLib)
  for(int i=0; i < m_image->Ptr()->width; ++i)
  {
    (*m_image)(0,i) = 0;
    (*m_image)(m_image->Ptr()->height-1, i) = 0;
  }

  for(int i=0; i < m_image->Ptr()->height; ++i)
  {
    (*m_image)(i,0) = 0;
    (*m_image)(i, m_image->Ptr()->width-1) = 0;
  }

  // extract blobs
  m_blobs = CBlobResult(m_image->Ptr(), NULL, threshold, false);

  // create a file with some of the extracted features
  //m_blobs.PrintBlobs( "output/blobs.txt" );

  // filter out all external blobs 
  // (i.e., connected components of pixels less than the threshold)
  m_blobs.Filter(m_blobs, B_INCLUDE, CBlobGetMean(), B_GREATER, 1);

  // create a file with filtered results
  //m_blobs.PrintBlobs("output/filteredBlobs.txt");
}

void ConnectedComponents::FilterMinArea(int area, CBlobResult& largeBlobs)
{
  // discard the blobs with less area than the specified area
  m_blobs.Filter(largeBlobs, B_INCLUDE, CBlobGetArea(), B_GREATER_OR_EQUAL, area);

  m_blobs = largeBlobs;
}

void ConnectedComponents::GetBlobImage(RgbImage& blobImage)
{
  blobImage.Clear();

  unsigned char r, g, b;
  srand(0);
  for(int i = 0; i < m_blobs.GetNumBlobs(); ++i)
  {
    r = rand() % 128 + 127;
    g = rand() % 128 + 127;
    b = rand() % 128 + 127;
    CBlob* blob = m_blobs.GetBlob(i);
    blob->FillBlob(blobImage.Ptr(), CV_RGB(r, g, b));
  }
}

void ConnectedComponents::SaveBlobImage(char* filename)
{
  IplImage *outputImage;
  outputImage = cvCreateImage(cvSize(m_image->Ptr()->width, m_image->Ptr()->height), IPL_DEPTH_8U, 3 );
  cvMerge(m_image, m_image, m_image, NULL, outputImage);

  unsigned char r, g, b;
  srand(0);
  for(int i = 0; i < m_blobs.GetNumBlobs(); ++i)
  {
    r = rand() % 256;
    g = rand() % 256;
    b = rand() % 256;
    CBlob* blob = m_blobs.GetBlob(i);
    blob->FillBlob(outputImage, CV_RGB(r, g, b));
  }

  if(!cvSaveImage(filename, outputImage)) 
    Error("Failed to save image.", "Error", 0);

  cvReleaseImage(&outputImage);
}

void ConnectedComponents::GetComponents(RgbImage& blobImage)
{
  blobImage.Clear();

  unsigned char r = 1;
  for(int i = 0; i < m_blobs.GetNumBlobs(); ++i)
  {
    CBlob* blob = m_blobs.GetBlob(i);
    blob->FillBlob(blobImage.Ptr(), CV_RGB(r, r, r));
    r++;
  }
}
void ConnectedComponents::FilterSaliency(BwImage& highThreshold, RgbImage& blobImg, float minSaliency,
                                         CBlobResult& salientBlobs, CBlobResult& unsalientBlobs)
{
  // calculate flow for each blob
  int numBlobs = m_blobs.GetNumBlobs();
  if(numBlobs == 0)
    return;

  float* salientPts = new float[numBlobs];
  for(int i = 0; i < numBlobs; ++i)
    salientPts[i] = 0.0f;

  for(int h = 0; h < blobImg.Ptr()->height; ++h)
  {
    for(int w = 0; w < blobImg.Ptr()->width; ++w)
    {
      unsigned char val = blobImg(h,w,0);

      if(val == 0)
        continue;

      if(highThreshold(h,w) != 0)
        salientPts[val-1] += 1.0f;
    }
  }

  // retain only blobs with high saliency
  CBlobResult blobs = m_blobs;
  m_blobs.ClearBlobs();
  salientBlobs.ClearBlobs();
  unsalientBlobs.ClearBlobs();
  for(int i = 0; i < numBlobs; ++i)
  {
    CBlob* blob = blobs.GetBlob(i);
    salientPts[i] = float(salientPts[i] / blob->Area());
    if(salientPts[i] >= minSaliency)
    {
      m_blobs.AddBlob(blobs.GetBlob(i));
      salientBlobs.AddBlob(blobs.GetBlob(i));
    }
    else
    {
      unsalientBlobs.AddBlob(blobs.GetBlob(i));
    }
  }

  delete[] salientPts;
}

void ConnectedComponents::ColorBlobs(IplImage* image, CBlobResult& blobs, CvScalar& color)
{
  for(int i = 0; i < blobs.GetNumBlobs(); ++i)
  {
    CBlob* blob = blobs.GetBlob(i);
    blob->FillBlob(image, color);
  }
}
