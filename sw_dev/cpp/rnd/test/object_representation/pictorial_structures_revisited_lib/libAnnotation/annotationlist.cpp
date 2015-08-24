
#include <libAnnotation/annotationlist.h>
#include <libAnnotation/xmlhelpers.h>
#include <string>  //-- [] 2012/08/07: Sang-Wook Lee
#include <cassert>

using namespace std;

//////////////////////////////////////////////////////////////////////////////
///
///
///   AnnotationList
///
///
//////////////////////////////////////////////////////////////////////////////

// AnnotationList::AnnotationList(const AnnotationList& other)
// {
//   m_vAnnotations.clear();
//   for (unsigned i=0; i<other.size(); i++)
//   {
//     this->addAnnotation(other.annotation(i));
//   }
// }

vector<string> AnnotationList::fileList() const
{
  vector<string> files;
  for(unsigned i=0; i<m_vAnnotations.size(); i++)
    files.push_back(m_vAnnotations[i].imageName());

  return files;
}

void AnnotationList::getFileList(vector<string>& files) const
{
  for(unsigned i=0; i<m_vAnnotations.size(); i++)
    files.push_back(m_vAnnotations[i].imageName());
}

void AnnotationList::printXML() const
{
  cout << "<annotationlist>\n";
  for(vector<Annotation>::const_iterator it=m_vAnnotations.begin(); it!=m_vAnnotations.end(); it++)
    it->printXML();
  cout << "</annotationlist>\n";
}

void AnnotationList::printIDL() const
{
  for(vector<Annotation>::const_iterator it=m_vAnnotations.begin(); it!=m_vAnnotations.end(); it++)
    it->printIDL();
}

void AnnotationList::save(const string& filename, bool bSaveRelativeToHome) const
{
  string::size_type pos = filename.rfind(".");
  if (pos == string::npos)
    return;

  else if (filename.substr(pos, string::npos).compare(".al")==0)
    saveXML(filename, bSaveRelativeToHome);
  else
    saveIDL(filename, bSaveRelativeToHome);
}

void AnnotationList::saveXML(const string& filename, bool bSaveRelativeToHome) const
{
  cerr << "AnnotationList::saveXML( " << filename << " )\n";

  ofstream f(filename.c_str());
  f << "<annotationlist>\n";
  for(vector<Annotation>::const_iterator it=m_vAnnotations.begin(); it!=m_vAnnotations.end(); it++)
    it->writeXML(f, bSaveRelativeToHome);
  f << "</annotationlist>\n";

  cerr << "Finished AnnotationList::save( " << filename << " )\n";
}

void AnnotationList::saveIDL(const string& filename, bool bSaveRelativeToHome) const
{
  cerr << "AnnotationList::saveIDL( " << filename << " )\n";
  ofstream f(filename.c_str());
  for(vector<Annotation>::const_iterator it=m_vAnnotations.begin(); it!=m_vAnnotations.end(); it++)
  {
    it->writeIDL(f, bSaveRelativeToHome);
    if (it+1!=m_vAnnotations.end())
      f << ";\n";
    else
      f << ".\n";
  }
}

void AnnotationList::load(const string& filename)
{
  //--- store path of anno file ---//
  m_sName=filename;
  string::size_type posSlash = filename.rfind("/");
  if (posSlash== string::npos)
    m_sPath = "";
  else
  {
    m_sName = filename.substr(posSlash+1,string::npos);
    m_sPath = filename.substr(0,posSlash+1);
  }

  //--- check for extension ---//
  string::size_type posExt = filename.rfind(".");
  string ext = filename.substr(posExt, string::npos);
  if (ext.compare(".idl")==0)
    loadIDL(filename);
  else if (ext.compare(".al")==0)
    loadXML(filename);
  else
    printf (" WARNING: No valid file extension provided: %s!!!\n", ext.c_str());

  //--- expand HOME if needed ---//
  
  if (m_vAnnotations.size() > 0) {
    string strHome = getenv("HOME");
    assert(strHome.length() > 0);

    for (int aidx = 0; aidx < (int)m_vAnnotations.size(); ++aidx) {
       if (m_vAnnotations[aidx].imageName()[0] == '~') {
         string strOldName = m_vAnnotations[aidx].imageName();

         string strNewName = strHome + strOldName.substr(1);
         m_vAnnotations[aidx].setImageName(strNewName);
       }
    }
  }

  return;
}


void AnnotationList::loadXML(const string& filename)
{
  cerr << "AnnotationList::loadXML( " << filename << " )\n";

  //--- read file ---//
  ifstream f(filename.c_str());
  string content;
  while (f.good())
  {
    content += f.get();
  }
  content.erase(content.length()-1,1);

  //--- chop in Annotations ---//
  vector<string> annoStrings = getElements("annotation", content);
  vector<string>::const_iterator it;
  for(it=annoStrings.begin(); it!=annoStrings.end(); it++)
  {
    Annotation a;
    a.parseXML(*it);
    m_vAnnotations.push_back(a);
  }

  cerr << "Finished AnnotationList::load( " << filename << " )\n";
}

void AnnotationList::loadIDL(const string& filename)
{
  cerr << "AnnotationList::loadIDL( " << filename << " )\n";

  //--- read file ---//
  ifstream f(filename.c_str());
  string content;
  while (f.good())
  {
    content += f.get();
  }
  content.erase(content.length()-1,1);

  //--- chop in Annotations ---//
  string annoString;
  vector<string> annoStrings;
  string::size_type pos=0, start=0, end;

  while (pos!=string::npos)
  {
    if (pos>0)
      start=pos+1;

    pos = content.find("\n", start);
    end=pos-1;

    annoString = content.substr(start,end-start);
    if (annoString.length()>0)
      annoStrings.push_back(annoString);
  }

  cerr << "Number of images: " << annoStrings.size() << endl;

  vector<string>::const_iterator it;
  for(it=annoStrings.begin(); it!=annoStrings.end(); it++)
  {
    Annotation a;
    a.parseIDL(*it, m_sPath);
    m_vAnnotations.push_back(a);
  }

  cerr << "Finished AnnotationList::loadIDL( " << filename << " )\n";
}

const Annotation& AnnotationList::findByName(const string& name, int frameNr) const
{
  vector<Annotation>::const_iterator it;
  for(it=m_vAnnotations.begin(); it!=m_vAnnotations.end(); it++)
  {
    string curName = it->imageName();
    int curFrameNr = it->frameNr();
    
    if (curName.compare(name)==0)
    	if (frameNr == -1)    
      		return *it;
     	else if (curFrameNr == frameNr)
      		return *it;      
  }
  return m_emptyAnnotation;
}

Annotation& AnnotationList::findByName(const string& name, int frameNr)
{
  vector<Annotation>::iterator it;
  for(it=m_vAnnotations.begin(); it!=m_vAnnotations.end(); it++)
  {
    string curName = it->imageName();
    int curFrameNr = it->frameNr();
    
    if (curName.compare(name)==0)
    	if (frameNr == -1)
      		return *it;
      	else if (curFrameNr == frameNr)
      		return *it;
  }
  return m_emptyAnnotation;
}

int AnnotationList::getIndexByName(const string& name, int frameNr) const
{
  for(unsigned i=0; i!=m_vAnnotations.size(); i++)
  {
    string curName = m_vAnnotations[i].imageName();
    int curFrameNr = m_vAnnotations[i].frameNr();
    
    if (curName.compare(name)==0)
    	if (frameNr == -1)
      		return i;
      	else if (curFrameNr == frameNr)
      		return i;   		
  }
  return -1;
}

void AnnotationList::removeAnnotationByName(const string& name, int frameNr)
{
  int idx = getIndexByName(name,frameNr);
  
  if (idx >= 0)
  	m_vAnnotations.erase(m_vAnnotations.begin()+idx);
}

void AnnotationList::addAnnotationByName(const string& name, int frameNr)
{
  if (getIndexByName(name, frameNr)==-1)
  {
    Annotation a(name);
    if (frameNr != -1)
    	a.setFrameNr(frameNr);
    m_vAnnotations.push_back(a);
  }
}

void AnnotationList::mergeAnnotationList(const AnnotationList& list)
{
  for(unsigned i=0; i<list.size(); i++)
    m_vAnnotations.push_back(list.annotation(i));
  //m_vAnnotations.insert(annos.begin(),annos.end());
}





