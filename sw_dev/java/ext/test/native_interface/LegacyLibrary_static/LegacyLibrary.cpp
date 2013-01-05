#include "LegacyLibrary.h"


namespace LegacyLibrary {

std::string & LegacyClass::get_property() { return property; }

const std::string & LegacyClass::get_property() const { return property; }

void LegacyClass::set_property(const std::string &prop) { this->property = prop; }

}  // namespace LegacyLibrary
