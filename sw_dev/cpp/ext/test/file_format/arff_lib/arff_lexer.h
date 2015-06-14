#ifndef __INCLUDED_ARFF_LEXER_H__
#define __INCLUDED_ARFF_LEXER_H__
/**
 * @file arff_lexer.h
 * @brief Contains the 'ArffLexer' class
 */


#include <string>

#include <arff_utils.h>
#include <arff_scanner.h>
#include <arff_token.h>


/**
 * @class ArffLexer arff_lexer.h
 * @brief Responsible for dividing the data inside 'arff' file into tokens
 */
class ArffLexer {
public:
    /**
     * @brief Constructor
     * @param _file file to be read
     */
    ArffLexer(const std::string& _file);

    /**
     * @brief Destructor
     */
    ~ArffLexer();

    /**
     * @brief Reads the next token and returns its type
     * @return token
     */
    ArffToken next_token();


    /** space */
    static const char SPACE;
    /** tab */
    static const char TAB;
    /** comment */
    static const char COMMENT;
    /** bracket open */
    static const char B_OPEN;
    /** bracket close */
    static const char B_CLOSE;
    /** single quote */
    static const char S_QUOTE;
    /** double quote */
    static const char D_QUOTE;
    /** comma */
    static const char COMMA;
    /** dealing with missing values */
    static const char MISS;


private:
    /**
     * @brief Helper function to read a string data
     * @return string data read
     *
     * This has the capability to read string with spaces
     * (when they are quoted)
     */
    std::string _read_str();

    /**
     * @brief Skips the comments line
     * @return true if the current line was a comment, else false
     *
     * Note that when this function returns, the 'current' character
     * will be newline, just before the beginning of the line.
     */
    bool _skip_comments();

    /**
     * @brief Skips the spaces
     *
     * Note that when this function returns, the 'current' character
     * will be a non-whitespace.
     */
    void _skip_spaces();

    /**
     * @brief Checks whether the input character is a whitespace or not
     * @param c character or interest
     * @return true if it is a whitespace, else false
     */
    bool _is_space(char c) const;

    /**
     * @brief Checks whether the input character is a comment or not
     * @param c character or interest
     * @return true if it is a comment, else false
     */
    bool _is_comment(char c) const;

    /**
     * @brief Checks whether the input character is bracket open or not
     * @param c character of interest
     * @return true if it is, else false
     */
    bool _is_bracket_open(char c) const;

    /**
     * @brief Checks whether the input character is bracket close or not
     * @param c character of interest
     * @return true if it is, else false
     */
    bool _is_bracket_close(char c) const;

    /**
     * @brief Checks whether the input character is single quote or not
     * @param c character of interest
     * @return true if it is, else false
     */
    bool _is_s_quote(char c) const;

    /**
     * @brief Checks whether the input character is double quote or not
     * @param c character of interest
     * @return true if it is, else false
     */
    bool _is_d_quote(char c) const;

    /**
     * @brief Checks whether the input character is comma or not
     * @param c character of interest
     * @return true if it is, else false
     */
    bool _is_comma(char c) const;

    /**
     * @brief Checks whether the input character is a '?' or not
     * @param c character of interest
     * @return true if it is, else false
     */
    bool _is_missing(char c) const;


    /** the scanner */
    ArffScanner* m_scanner;
    /** variable to catch for '}' occurences */
    bool m_b_close;
};


/* DO NOT WRITE ANYTHING BELOW THIS LINE!!! */
#endif // __INCLUDED_ARFF_LEXER_H__
