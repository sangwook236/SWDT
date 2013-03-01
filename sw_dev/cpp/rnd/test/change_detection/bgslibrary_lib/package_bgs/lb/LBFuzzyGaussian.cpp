#include "LBFuzzyGaussian.h"

LBFuzzyGaussian::LBFuzzyGaussian() : firstTime(true), showOutput(true), sensitivity(72), bgThreshold(162), learningRate(49), noiseVariance(195)
{
  std::cout << "LBFuzzyGaussian()" << std::endl;
}

LBFuzzyGaussian::~LBFuzzyGaussian()
{
  std::cout << "~LBFuzzyGaussian()" << std::endl;
}

void LBFuzzyGaussian::process(const cv::Mat &img_input, cv::Mat &img_output)
{
  if(img_input.empty())
    return;

  loadConfig();
  
  IplImage *frame = new IplImage(img_input);
  
  if(firstTime)
  {
    saveConfig();

    int w = cvGetSize(frame).width;
    int h = cvGetSize(frame).height;

    m_pBGModel = new BGModelFuzzyGauss(w,h);
    m_pBGModel->InitModel(frame);
  }
  
  m_pBGModel->setBGModelParameter(0,sensitivity);
  m_pBGModel->setBGModelParameter(1,bgThreshold);
  m_pBGModel->setBGModelParameter(2,learningRate);
  m_pBGModel->setBGModelParameter(3,noiseVariance);

  m_pBGModel->UpdateModel(frame);

  img_foreground = cv::Mat(m_pBGModel->GetFG());
  img_background = cv::Mat(m_pBGModel->GetBG());
    
  if(showOutput)
  {
    cv::imshow("FG Mask", img_foreground);
    cv::imshow("FG Model", img_background);
  }

  img_foreground.copyTo(img_output);
  
  delete frame;
  
  firstTime = false;
}

void LBFuzzyGaussian::finish(void)
{
  delete m_pBGModel;
}

void LBFuzzyGaussian::saveConfig()
{
  CvFileStorage* fs = cvOpenFileStorage("./config/LBFuzzyGaussian.xml", 0, CV_STORAGE_WRITE);

  cvWriteInt(fs, "sensitivity", sensitivity);
  cvWriteInt(fs, "bgThreshold", bgThreshold);
  cvWriteInt(fs, "learningRate", learningRate);
  cvWriteInt(fs, "noiseVariance", noiseVariance);
  
  cvWriteInt(fs, "showOutput", showOutput);

  cvReleaseFileStorage(&fs);
}

void LBFuzzyGaussian::loadConfig()
{
  CvFileStorage* fs = cvOpenFileStorage("./config/LBFuzzyGaussian.xml", 0, CV_STORAGE_READ);
  
  sensitivity = cvReadIntByName(fs, 0, "sensitivity", 72);
  bgThreshold = cvReadIntByName(fs, 0, "bgThreshold", 162);
  learningRate = cvReadIntByName(fs, 0, "learningRate", 49);
  noiseVariance = cvReadIntByName(fs, 0, "noiseVariance", 195);
  
  showOutput = cvReadIntByName(fs, 0, "showOutput", true);

  cvReleaseFileStorage(&fs);
}
