#ifndef __INCLUDED_ARFF_UTILS_H__
#define __INCLUDED_ARFF_UTILS_H__
/**
 * @file arff_utils.h
 * @brief Contains some utility functions used in this project.
 */

#include <stdarg.h>

#include <string>
#include <sstream>


/** 32 bit integer */
typedef long int    int32;
/** 64 bit integer */
typedef long long   int64;


/**
 * @brief Handy function to throw std::exception
 * @param file file where the error is occuring
 * @param line line where the error is occuring
 * @param fmt format of the error message (followed by the args)
 */
void throw_ex(const char* file, int64 line, const char* fmt, ...);


/** Handy macro to throw exceptions */
#define THROW(fmt, ...)                                         \
    throw_ex(__FILE__, (int64)__LINE__, fmt, ##__VA_ARGS__)


/**
 * @brief Converts a number to string
 * @param num the number to be converted
 * @return the desired string
 */
template <typename T>
std::string num2str(T num) {
    std::string str;
    std::ostringstream oss;
    oss << num;
    str = oss.str();
    return str;
}


/**
 * @brief Converts a string to number
 * @param str the string to be converted
 * @param num the desired number
 */
template <typename T>
void str2num(const std::string& str, T& num) {
    std::istringstream iss(str);
    iss >> num;
    if(iss.fail()) {
        THROW("(str2num) Input '%s' is not a number!", str.c_str());
    }
    return;
}


/**
 * @brief Convert the input character to lower case
 * @param c input character
 * @return lowercase version of the input character
 */
char to_lower(char c);


/**
 * @brief Performs a case-insensitive match against the 2 strings
 * @param str the input string
 * @param ref the reference string
 * @return true if the 2 strings match, else false
 */
bool icompare(const std::string& str, const std::string& ref);


/* DO NOT WRITE ANYTHING BELOW THIS LINE!!! */
#endif // __INCLUDED_ARFF_UTILS_H__
