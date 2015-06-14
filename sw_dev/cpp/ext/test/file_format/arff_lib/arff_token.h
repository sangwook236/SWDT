#ifndef __INCLUDED_ARFF_TOKEN_H__
#define __INCLUDED_ARFF_TOKEN_H__
/**
 * @file arff_token.h
 * @brief Contains the 'ArffToken' class
 */

#include <string>

#include <arff_utils.h>


/**
 * @enum ArffTokenEnum
 * @brief Tokens in the ARFF file
 */
enum ArffTokenEnum {
    /** relation tag */
    RELATION = 0,
    /** attribute tag */
    ATTRIBUTE,
    /** data tag */
    DATA_TOKEN,
    /** missing */
    MISSING_TOKEN,
    /** numeric type */
    NUMERIC_TOKEN,
    /** string type */
    STRING_TOKEN,
    /** date type */
    DATE_TOKEN,
    /** value token (could be name, nominal, value) */
    VALUE_TOKEN,
    /** flower bracket open */
    BRKT_OPEN,
    /** flower bracket close */
    BRKT_CLOSE,
    /** end of file has been reached */
    END_OF_FILE,
    /** unknown type (usually an error) */
    UNKNOWN_TOKEN,
};

/**
 * @brief Utility function to convert ArffTokenEnum to string
 * @param type the enum value to be converted
 * @return desired string
 */
std::string arff_token2str(ArffTokenEnum type);


/**
 * @struct ArffToken arff_token.h
 * @brief An ARFF Token read by the ArffLexer class
 */
struct ArffToken {
public:
    /**
     * @brief Constructor
     * @param _str the token string
     * @param _token the token enum
     */
    ArffToken(const std::string& _str, ArffTokenEnum _token);

    /**
     * @brief Copy constructor
     * @param _src source object to copy from
     */
    ArffToken(const ArffToken& _src);

    /**
     * @brief Destructor
     */
    ~ArffToken();

    /**
     * @brief token string
     * @return string
     */
    std::string token_str() const;

    /**
     * @brief token enum
     * @return enum
     */
    ArffTokenEnum token_enum() const;

    /**
     * @brief token value as a 32b integer
     * @return integer
     *
     * Note, this function is meaningful only if the token is of
     * 'numeric' type!
     */
    int32 token_int32() const;

    /**
     * @brief token value as a 64b integer
     * @return integer
     *
     * Note, this function is meaningful only if the token is of
     * 'numeric' type!
     */
    int64 token_int64() const;

    /**
     * @brief token value as a float
     * @return float
     *
     * Note, this function is meaningful only if the token is of
     * 'numeric' type!
     */
    float token_float() const;

    /**
     * @brief token value as a double
     * @return double
     *
     * Note, this function is meaningful only if the token is of
     * 'numeric' type!
     */
    double token_double() const;


private:
    /** string */
    std::string m_str;
    /** enum */
    ArffTokenEnum m_enum;
};


/* DO NOT WRITE ANYTHING BELOW THIS LINE!!! */
#endif // __INCLUDED_ARFF_TOKEN_H__
