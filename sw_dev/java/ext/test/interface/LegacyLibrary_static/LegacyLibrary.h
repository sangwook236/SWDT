#include <string>

namespace LegacyLibrary {

class LegacyClass
{
public:
    std::string & get_property();
    const std::string & get_property() const;
    void set_property(const std::string &prop);

    std::string property;
};

}  // namespace LegacyLibrary
