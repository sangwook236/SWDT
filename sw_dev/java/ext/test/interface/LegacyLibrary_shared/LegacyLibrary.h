#ifdef LEGACYLIBRARY_EXPORTS
#define LEGACYLIBRARY_API __declspec(dllexport)
#else
#define LEGACYLIBRARY_API __declspec(dllimport)
#endif

#include <string>

namespace LegacyLibrary {

class LEGACYLIBRARY_API LegacyClass
{
public:
    std::string & get_property();
    const std::string & get_property() const;
    void set_property(const std::string &prop);

    std::string property;
};

}  // namespace LegacyLibrary
