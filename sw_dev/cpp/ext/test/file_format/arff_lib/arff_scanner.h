#ifndef __INCLUDED_ARFF_SCANNER_H__
#define __INCLUDED_ARFF_SCANNER_H__
/**
 * @file arff_scanner.h
 * @brief Contains the 'ArffScanner' class
 */


#include <stdio.h>

#include <string>

#include <arff_utils.h>


/**
 * @class ArffScanner arff_scanner.h
 * @brief Class responsible for reading the 'arff' file
 *
 * This class assumes linux-style newlines!
 */
class ArffScanner {
public:
    /**
     * @brief Constructor
     * @param _file file to be read
     */
    ArffScanner(const std::string& _file);

    /**
     * @brief Destructor
     */
    ~ArffScanner();

    /**
     * @brief Return the next character in the stream
     * @return character
     */
    char next();

    /**
     * @brief Returns the currently read char from the file
     * @return current character
     */
    char current() const;

    /**
     * @brief Returns the previously read char from the file
     * @return previous character
     */
    char previous() const;

    /**
     * @brief Returns the current line position
     * @return current line
     */
    int64 line() const;

    /**
     * @brief Returns the current column position
     * @return current column
     */
    int64 column() const;

    /**
     * @brief Whether the file has reached end or not
     * @return true if end-of-file, else false
     */
    bool eof() const;

    /**
     * @brief Give a nice error message along with file,line,col info
     * @param msg actual error message to be prepended with the above info
     * @return prepended 'meaningful' error message
     */
    std::string err_msg(const std::string& msg) const;

    /**
     * @brief Checks whether the given character is newline or not
     * @param c the character
     * @return true if the character is newline, else false
     */
    bool is_newline(char c) const;


    /** new-line character */
    static const char NEWLINE;


private:


    /** file being read */
    std::string m_file;
    /** current line being read */
    int64 m_line;
    /** current position in the row being read */
    int64 m_col;
    /** current character read from the file */
    char m_char;
    /** previous character read from the file */
    char m_prev_char;
    /** file pointer */
    FILE* m_fp;
};


/* DO NOT WRITE ANYTHING BELOW THIS LINE!!! */
#endif // __INCLUDED_ARFF_SCANNER_H__
