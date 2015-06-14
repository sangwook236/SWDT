#ifndef __INCLUDED_ARFF_PARSER_H__
#define __INCLUDED_ARFF_PARSER_H__
/**
 * @file arff_parser.h
 * @brief Contains class 'ArffParser'
 */

#include <string>

#include <arff_lexer.h>
#include <arff_data.h>


/**
 * @class ArffParser arff_parser.h
 * @brief Main class for parsing ARFF files
 */
class ArffParser {
public:
    /**
     * @brief Constructor
     * @param _file File to be parsed
     */
    ArffParser(const std::string& _file);

    /**
     * @brief Destructor
     */
    ~ArffParser();

    /**
     * @brief Main function for parsing the file
     * @return the 'ArffData' object after parsing the file
     *
     * Note that this pointer will still be owned by this class!
     */
    ArffData* parse();


private:
    /**
     * @brief Reads the 'relation' token
     */
    void _read_relation();

    /**
     * @brief Reads the attributes
     */
    void _read_attrs();

    /**
     * @brief Reads one attribute
     */
    void _read_attr();

    /**
     * @brief Reads the data
     */
    void _read_instances();


    /** lexer for generating tokens */
    ArffLexer* m_lexer;
    /** whether you have already parsed the file or not */
    bool m_parsed;
    /** the data parsed from the ARFF file */
    ArffData* m_data;
};


/* DO NOT WRITE ANYTHING BELOW THIS LINE!!! */
#endif // __INCLUDED_ARFF_PARSER_H__
