
#include <string>  //-- [] 2012/08/07: Sang-Wook Lee
#include <libAnnotation/xmlhelpers.h>

using namespace std;

//////////////////////////////////////////////////////////////////////////////
///
///
///   XML helpers
///
///
//////////////////////////////////////////////////////////////////////////////

string getElementDataString(const string& name, const string& doc)
{
  
  //cout << "  getElementDataInt()"<< endl;
  //cout << "    TagName: "<< name <<" Doc: " << doc << endl;
  
  string tagStart = string("<")+name+">";
  string tagEnd = string("</")+name+">";
  
  unsigned start = tagStart.length();
  unsigned end = doc.find(tagEnd, start);
  
  string dataString = doc.substr(start,end-start);
  //cout << "    Data: " << dataString << "-> " << atoi(dataString.c_str()) << endl;
  
  return dataString;
  
}

int getElementDataInt(const string& name, const string& doc)
{
  
  //cout << "  getElementDataInt()"<< endl;
  //cout << "    TagName: "<< name <<" Doc: " << doc << endl;
  
  string tagStart = string("<")+name+">";
  string tagEnd = string("</")+name+">";
  
  unsigned start = tagStart.length();
  unsigned end = doc.find(tagEnd, start);
  
  string dataString = doc.substr(start,end-start);
  //cout << "    Data: " << dataString << "-> " << atoi(dataString.c_str()) << endl;
  
  return atoi(dataString.c_str());
  
}

float getElementDataFloat(const string& name, const string& doc)
{
  
  //cout << "  getElementDataInt()"<< endl;
  //cout << "    TagName: "<< name <<" Doc: " << doc << endl;
  
  string tagStart = string("<")+name+">";
  string tagEnd = string("</")+name+">";
  
  unsigned start = tagStart.length();
  unsigned end = doc.find(tagEnd, start);
  
  string dataString = doc.substr(start,end-start);
  //cout << "    Data: " << dataString << "-> " << atof(dataString.c_str()) << endl;
  
  return atof(dataString.c_str());
  
}

vector<string> getElements(const string& name, const string& doc)
{
  
  //cout << "  getElements(" << name << ")"<< endl;
  //cout << "    Doc: " << doc << endl << "    End Doc" << endl;
  
  string::size_type start, end, pos=0;
  string elementString;
  vector<string> elementStrings;
  
  string tagStart = string("<")+name+">";
  string tagEnd = string("</")+name+">";
  
  while (pos!=string::npos)
  {
    
    pos = doc.find(tagStart, pos);
    if (pos==string::npos) break;
    start = pos;
    
    pos = doc.find(tagEnd, pos+1);
    if (pos==string::npos) break;
    end = pos;
    
    elementString = doc.substr(start,end-start+tagEnd.length());
    elementStrings.push_back(elementString);
    
    //cout << "    New Element found: " << endl;
    //cout << elementString << endl;
    //cout << "    End new Element!" << endl;
    
  }  
  
  return elementStrings;
  
}


