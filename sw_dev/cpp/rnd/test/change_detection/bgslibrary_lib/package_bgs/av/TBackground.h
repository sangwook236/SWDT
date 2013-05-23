#pragma once
/*
*  TBackground.h
*  Framework
*
*  Created by Robinault Lionel on 07/12/11.
*
*/
#include <iostream>
#include <cv.h>
#include <highgui.h>

class TBackground
{
public:
  TBackground(void);
  virtual ~TBackground(void);

  virtual void Clear(void);
  virtual void Reset(void);

  virtual int UpdateBackground(IplImage * pSource, IplImage *pBackground, IplImage *pMotionMask);
  virtual int UpdateTest(IplImage *pSource, IplImage *pBackground, IplImage *pTest, int nX, int nY, int nInd);
  virtual IplImage *CreateTestImg();

  virtual int GetParameterCount(void);
  virtual std::string GetParameterName(int nInd);
  virtual std::string GetParameterValue(int nInd);
  virtual int SetParameterValue(int nInd, std::string csNew);

protected:
  virtual int Init(IplImage * pSource);
  virtual bool isInitOk(IplImage * pSource, IplImage *pBackground, IplImage *pMotionMask);
};
