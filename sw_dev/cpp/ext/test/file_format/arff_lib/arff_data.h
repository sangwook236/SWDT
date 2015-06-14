#ifndef __INCLUDED_ARFF_DATA_H__
#define __INCLUDED_ARFF_DATA_H__
/**
 * @file arff_data.h
 * @brief Contains the 'ArffData' class
 */

#include <string>
#include <vector>
#include <map>

#include <arff_utils.h>
#include <arff_instance.h>
#include <arff_attr.h>


/** nominal values */
typedef std::vector<std::string> ArffNominal;
/** date formats */
typedef std::map<std::string, std::string> ArffDateFormat;


/**
 * @class ArffData arff_data.h
 * @brief Class to represent the data parsed from the ARFF files
 */
class ArffData {
public:
    /**
     * @brief Constructor
     */
    ArffData();

    /**
     * @brief Destructor
     */
    ~ArffData();

    /**
     * @brief Set the relation name
     * @param name the name
     */
    void set_relation_name(const std::string& name);

    /**
     * @brief Name of the relation for this ARFF file
     * @return the name
     */
    std::string get_relation_name() const;

    /**
     * @brief Number of attributes
     * @return number
     */
    int32 num_attributes() const;

    /**
     * @brief Add an attribute
     * @param attr attribute pointer
     *
     * Note that this pointer will be owned by this class from here onwards!
     */
    void add_attr(ArffAttr* attr);

    /**
     * @brief Get attribute pointer at the given location
     * @param idx location (Starts from 0)
     * @return pointer
     *
     * Note that this pointer will still be owned by this class!
     */
    ArffAttr* get_attr(int32 idx) const;

    /**
     * @brief Number of instances
     * @return number
     */
    int32 num_instances() const;

    /**
     * @brief Add an instance
     * @param inst instance pointer
     *
     * Note that this pointer will be owned by this class from here onwards!
     */
    void add_instance(ArffInstance* inst);

    /**
     * @brief Get instance pointer at the given location
     * @param idx location (Starts from 0)
     * @return pointer
     *
     * Note that this pointer will still be owned by this class!
     */
    ArffInstance* get_instance(int32 idx) const;

    /**
     * @brief Add a nominal value to the list
     * @param name name of the nominal list
     * @param val nominal value
     */
    void add_nominal_val(const std::string& name, const std::string& val);

    /**
     * @brief Get a nominal list
     * @param name name of the nominal list
     * @return list
     */
    ArffNominal get_nominal(const std::string& name);

    /**
     * @brief Add a date format
     * @param name name of the date data
     * @param val date format
     */
    void add_date_format(const std::string& name, const std::string& val);

    /**
     * @brief Get a date format
     * @param name name of the date data
     * @return format
     */
    std::string get_date_format(const std::string& name);

    /**
     * @brief Prepare an ARFF file from this object
     * @param file file to be written to
     */
    void write_arff(const std::string& file);


private:
    /**
     * @brief Cross check the attributes against the given instance
     * @param inst instance pointer
     */
    void _cross_check_instance(ArffInstance* inst);


    /** relation name */
    std::string m_rel;
    /** nominal values */
    std::map<std::string, ArffNominal > m_nominals;
    /** date formats */
    ArffDateFormat m_formats;
    /** number of attributes */
    int32 m_num_attrs;
    /** attributes */
    std::vector<ArffAttr*> m_attrs;
    /** number of instances */
    int32 m_num_instances;
    /** instances */
    std::vector<ArffInstance*> m_instances;
};


/* DO NOT WRITE ANYTHING BELOW THIS LINE!!! */
#endif // __INCLUDED_ARFF_DATA_H__
