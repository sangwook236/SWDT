#include <boost/property_map/property_map.hpp>
#include <boost/property_map/dynamic_property_map.hpp>
#include <iostream>


namespace {
namespace local {

template <typename AddressMap>
void foo(AddressMap &address)
{
	typedef typename boost::property_traits<AddressMap>::value_type value_type;
	typedef typename boost::property_traits<AddressMap>::key_type key_type;

	const key_type fred = "Fred";

	const value_type &old_address = boost::get(address, fred);
	const value_type new_address = "384 Fitzpatrick Street";
	boost::put(address, fred, new_address);

	const key_type joe = "Joe";
	value_type &joes_address = address[joe];
	joes_address = "325 Cushing Avenue";
}

void static_property_map()
{
	std::map<std::string, std::string> name2address;
	boost::associative_property_map<std::map<std::string, std::string> > address_map(name2address);

	name2address.insert(std::make_pair(std::string("Fred"), std::string("710 West 13th Street")));
	name2address.insert(std::make_pair(std::string("Joe"), std::string("710 West 13th Street")));

	foo(address_map);

	for (std::map<std::string, std::string>::iterator i = name2address.begin(); i != name2address.end(); ++i)
		std::cout << i->first << ": " << i->second << std::endl;
}

#if 0
template <typename AgeMap, typename GPAMap>
void manipulate_freds_info(AgeMap &ages, GPAMap &gpas)
{
	typedef typename boost::property_traits<AgeMap>::key_type name_type;
	typedef typename boost::property_traits<AgeMap>::value_type age_type;
	typedef typename boost::property_traits<GPAMap>::value_type gpa_type;

	const name_type fred = "Fred";

	const age_type old_age = boost::get(ages, fred);
	const gpa_type old_gpa = boost::get(gpas, fred);

	std::cout << "Fred's old age: " << old_age << "\n" << "Fred's old gpa: " << old_gpa << "\n";

	const age_type new_age = 18;
	const gpa_type new_gpa = 3.9;
	boost::put(ages, fred, new_age);
	boost::put(gpas, fred, new_gpa);
}
#else
void manipulate_freds_info(boost::dynamic_properties &properties)
{
	const std::string fred = "Fred";

	const int old_age = boost::get<int>("age", properties, fred);
	const std::string old_gpa = boost::get("gpa", properties, fred);

	std::cout << "Fred's old age: " << old_age << "\nFred's old gpa: " << old_gpa << std::endl;

	const std::string new_age = "18";
	const double new_gpa = 3.9;
	boost::put("age", properties, fred, new_age);
	boost::put("gpa", properties, fred, new_gpa);
}
#endif

void dynamic_property_map()
{
	// build property maps using associative_property_map
	std::map<std::string, int> name2age;
	std::map<std::string, double> name2gpa;
	boost::associative_property_map<std::map<std::string, int> > age_map(name2age);
	boost::associative_property_map<std::map<std::string, double> > gpa_map(name2gpa);

	const std::string fred("Fred");
	// add key-value information
	name2age.insert(std::make_pair(fred, 17));
	name2gpa.insert(std::make_pair(fred, 2.7));

	// build and populate dynamic interface
	boost::dynamic_properties properties;
	properties.property("age", age_map);
	properties.property("gpa", gpa_map);

	manipulate_freds_info(properties);

	std::cout << "Fred's age: " << boost::get(age_map, fred) << "\nFred's gpa: " << boost::get(gpa_map, fred) << std::endl;
}

}  // local
}  // unnamed namespace

void property_map()
{
	local::static_property_map();
	local::dynamic_property_map();
}
