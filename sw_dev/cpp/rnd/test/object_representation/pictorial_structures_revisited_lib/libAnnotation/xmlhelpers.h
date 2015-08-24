
#ifndef XMLHELPERS_H
#define XMLHELPERS_H

#include <vector>
#include <iostream>

std::string getElementDataString(const std::string& name, const std::string& doc);
int getElementDataInt(const std::string& name, const std::string& doc);
float getElementDataFloat(const std::string& name, const std::string& doc);
std::vector<std::string> getElements(const std::string& name, const std::string& doc);


#endif

