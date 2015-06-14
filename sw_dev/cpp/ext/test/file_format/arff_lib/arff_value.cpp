#include <arff_value.h>


std::string arff_value2str(ArffValueEnum e) {
    switch(e) {
    case INTEGER: return "INTEGER";
    case FLOAT:   return "FLOAT";
    case DATE:    return "DATE";
    case STRING:  return "STRING";
    case NUMERIC: return "NUMERIC";
    case NOMINAL: return "NOMINAL";
    default:      return "UNKNOWN";
    }
}




ArffValue::ArffValue(int32 i/*=0*/): m_int(i),
                                     m_float(0.0),
                                     m_str(""),
                                     m_type(INTEGER),
                                     m_missing(false) {
}

ArffValue::ArffValue(float f): m_int(0),
                               m_float(f),
                               m_str(""),
                               m_type(FLOAT),
                               m_missing(false) {
}

ArffValue::ArffValue(const std::string& str, bool convert/*=true*/,
                     bool is_date/*=false*/): m_int(0),
                                              m_float(0.0f),
                                              m_str(str),
                                              m_missing(false) {
    m_type = (is_date)? DATE : STRING;
    if(convert && (m_type == STRING)) {
        // try float
        try {
            float f;
            str2num<float>(str, f);
            m_type = FLOAT;
            m_float = f;
        }
        catch(...) {
            // else leave it as STRING
        }
    }
}

ArffValue::ArffValue(const ArffValue& src) : m_int(src.m_int),
                                             m_float(src.m_float),
                                             m_str(src.m_str),
                                             m_type(src.m_type),
                                             m_missing(false) {
}

ArffValue::ArffValue(ArffValueEnum type) : m_int(0),
                                           m_float(0.0f),
                                           m_str(""),
                                           m_type(type),
                                           m_missing(true) {
}

ArffValue::~ArffValue() {
}

ArffValueEnum ArffValue::type() const {
    return m_type;
}

void ArffValue::set(int32 i) {
    m_type = INTEGER;
    m_int = i;
}

void ArffValue::set(float f) {
    m_type = FLOAT;
    m_float = f;
}

void ArffValue::set(const std::string& str, ArffValueEnum e/*=STRING*/) {
    if((e != DATE) && (e != STRING)) {
        THROW("%s expects 'DATE' or 'STRING', you've passed '%s'!",
              "ArffValue::set", arff_value2str(e).c_str());
    }
    m_type = e;
    m_str = str;
}

bool ArffValue::missing() const {
    return m_missing;
}

ArffValue::operator int32() const {
    switch(m_type) {
    case INTEGER:
        return m_int;
    case FLOAT:
        return (int32)m_float;
    default:
        THROW("operator int32 cannot work on type '%s'!",
              arff_value2str(m_type).c_str());
        return 0;  // I need to keep the compiler happy!
    }
}

ArffValue::operator float() const {
    switch(m_type) {
    case INTEGER:
        return (float)m_int;
    case FLOAT:
        return m_float;
    default:
        THROW("operator float cannot work on type '%s'!",
              arff_value2str(m_type).c_str());
        return 0.0f;  // I need to keep the compiler happy!
    }
}

ArffValue::operator std::string() const {
    switch(m_type) {
    case INTEGER:
        return num2str<int32>(m_int);
    case FLOAT:
        return num2str<float>(m_float);
    case DATE:
    case STRING:
        return m_str;
    default:
        THROW("operator std::string cannot work on type '%s'!",
              arff_value2str(m_type).c_str());
        return "";  // I need to keep the compiler happy!
    }
}

bool ArffValue::operator ==(const ArffValue& right) const {
    if(m_type != right.m_type) {
        return false;
    }
    switch(m_type) {
    case INTEGER:
        return (m_int == right.m_int);
    case FLOAT:
        return (m_float == right.m_float);
    case DATE:
    case STRING:
        return (m_str == right.m_str);
    default:
        return false;
    }
}

bool ArffValue::operator ==(int32 right) const {
    if(m_type != INTEGER) {
        return false;
    }
    return m_int == right;
}

bool ArffValue::operator ==(float right) const {
    if(m_type != FLOAT) {
        return false;
    }
    return m_float == right;
}

bool ArffValue::operator ==(const std::string& right) const {
    if((m_type != DATE) && (m_type != STRING)) {
        return false;
    }
    return m_str == right;
}

bool operator ==(int32 left, const ArffValue& right) {
    return (right == left);
}

bool operator ==(float left, const ArffValue& right) {
    return (right == left);
}

bool operator ==(const std::string& left, const ArffValue& right) {
    return (right == left);
}
