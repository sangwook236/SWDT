#ifndef __INCLUDED_ARFF_ATTR_H__
#define __INCLUDED_ARFF_ATTR_H__
/**
 * @file arff_attr.h
 * @brief Contains 'ArffAttr' class
 */

#include <string>

#include <arff_value.h>


/**
 * @class ArffAttr arff_attr.h
 * @brief Class to represent an ARFF attribute
 */
class ArffAttr {
public:
    /**
     * @brief Constructor
     * @param name name of this attribute
     * @param type attribute type
     */
    ArffAttr(const std::string& name, ArffValueEnum type);

    /**
     * @brief Destructor
     */
    ~ArffAttr();

    /**
     * @brief Name of this attribute
     * @return name
     */
    std::string name() const;

    /**
     * @brief Type of this attribute
     * @return type
     */
    ArffValueEnum type() const;


private:
    /** attribute name */
    std::string m_name;
    /** attribute type */
    ArffValueEnum m_enum;
};


/* DO NOT WRITE ANYTHING BELOW THIS LINE!!! */
#endif // __INCLUDED_ARFF_ATTR_H__
