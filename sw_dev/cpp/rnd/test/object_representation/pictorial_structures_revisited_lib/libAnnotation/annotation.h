
#ifndef ANNOTATION_H
#define ANNOTATION_H

#include <vector>
#include <iostream>
#include <fstream>

#include <libAnnotation/annorect.h>


class Annotation
{
public:
  Annotation() : m_nImageWidth(0), m_nImageHeight(0), m_bIsTiffStream(false) {};
  Annotation(const std::string& name): m_nImageWidth(0), m_nImageHeight(0), m_bIsTiffStream(false)  {m_sName=name;};
  //Annotation(const Annotation& other);
  ~Annotation(){};


private:
  // new 
  int m_nImageWidth;
  int m_nImageHeight;

  std::string m_sPath;
  std::string m_sName;
  std::string m_sDim;
  int m_dFrameNr;
  bool m_bIsTiffStream;
  std::vector<AnnoRect> m_vRects;
  
public:
  int imageWidth() {return m_nImageWidth;}
  int imageHeight() {return m_nImageHeight;}
  void setImageWidth(int nImageWidth) { m_nImageWidth = nImageWidth; }
  void setImageHeight(int nImageHeight) { m_nImageHeight = nImageHeight; }

  unsigned size() const {return m_vRects.size();};
  void clear() {m_vRects.clear();};

  void sortByScore();

  const AnnoRect& annoRect(unsigned i) const {return m_vRects[i];};
  AnnoRect& annoRect(unsigned i) {return m_vRects[i];};

  const AnnoRect& operator[]  (unsigned i) const {return m_vRects[i];};
  AnnoRect& operator[] (unsigned i) {return m_vRects[i];};
  
  void addAnnoRect(const AnnoRect& rect) {m_vRects.push_back(rect);};
  void removeAnnoRect(unsigned pos) {m_vRects.erase(m_vRects.begin()+pos);};

  const std::string& imageName() const {return m_sName;};  
  void setImageName(const std::string& name) {m_sName=name;};
  const std::string& imagePath() const {return m_sPath;};
  void setPath(const std::string& path) {m_sPath=path;};
  
  std::string fileName() {
  	std::string::size_type pos = m_sName.rfind("/");  	
  	if (pos != std::string::npos)
  		return m_sName.substr(pos+1);
  	else
  		return std::string("");
  }
  
  const std::string& imageDim() const {return m_sDim;};
  void setDim(const std::string& dim) {m_sDim=dim;};
  
  //Stream extensions
  const int frameNr() const {return m_dFrameNr;}
  void setFrameNr(const int frameNr) {m_dFrameNr = frameNr; m_bIsTiffStream = true;}
  const bool isStream() const {return m_bIsTiffStream;}
  
  //--- IO ---//
  void printXML() const;
  void printIDL() const;
  
  void writeXML(std::ofstream&, bool bSaveRelativeToHome) const;
  void writeIDL(std::ofstream&, bool bSaveRelativeToHome) const;

  void parseXML(const std::string&);
  void parseIDL(const std::string& annostring, const std::string& sPath="");

};


#endif
