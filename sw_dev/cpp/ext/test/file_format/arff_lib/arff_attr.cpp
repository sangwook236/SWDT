#include <arff_attr.h>




ArffAttr::ArffAttr(const std::string& name, ArffValueEnum type):
    m_name(name),
    m_enum(type) {
}

ArffAttr::~ArffAttr() {
}

std::string ArffAttr::name() const {
    return m_name;
}

ArffValueEnum ArffAttr::type() const {
    return m_enum;
}
