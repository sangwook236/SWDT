#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/global_fun.hpp>
#include <boost/multi_index/mem_fun.hpp>
#include <iostream>


namespace {
namespace local {

struct employee
{
	int id;
	std::string name;
	int age;

	employee(int id_, std::string name_, int age_)
	: id(id_), name(name_), age(age_)
	{}

	friend std::ostream & operator<<(std::ostream &os, const employee &e)
	{
		os << e.id << " " << e.name << " " << e.age << std::endl;
		return os;
	}
};

/* tags for accessing the corresponding indices of employee_set */

struct id {};
struct name {};
struct age {};

/* see Compiler specifics: Use of member_offset for info on
 * BOOST_MULTI_INDEX_MEMBER
 */

/* Define a multi_index_container of employees with following indices:
 *   - a unique index sorted by employee::id,
 *   - a non-unique index sorted by employee::name,
 *   - a non-unique index sorted by employee::age.
 */

typedef boost::multi_index_container<
	employee,
	boost::multi_index::indexed_by<
		boost::multi_index::ordered_unique<boost::multi_index::tag<id>, BOOST_MULTI_INDEX_MEMBER(employee, int, id)>,
		boost::multi_index::ordered_non_unique<boost::multi_index::tag<name>, BOOST_MULTI_INDEX_MEMBER(employee, std::string, name)>,
		boost::multi_index::ordered_non_unique<boost::multi_index::tag<age>, BOOST_MULTI_INDEX_MEMBER(employee, int, age)>
	>
> employee_set;

template<typename Tag, typename MultiIndexContainer>
//void print_out_by(const MultiIndexContainer &s, Tag * = NULL)  // fixes a MSVC++ 6.0 bug with implicit template function params
void print_out_by(const MultiIndexContainer &s)
{
	// obtain a reference to the index tagged by Tag
	const typename boost::multi_index::index<MultiIndexContainer, Tag>::type &i = boost::multi_index::get<Tag>(s);

	typedef typename MultiIndexContainer::value_type value_type;

	// dump the elements of the index to cout
	std::copy(i.begin(), i.end(), std::ostream_iterator<value_type>(std::cout));
}

void basic()
{
	employee_set es;

	es.insert(employee(0, "Joe", 31));
	es.insert(employee(1, "Robert", 27));
	es.insert(employee(2, "John", 40));

	// next insertion will fail, as there is an employee with the same ID
	es.insert(employee(2, "Aristotle", 2387));

	es.insert(employee(3, "Albert", 20));
	es.insert(employee(4, "John", 57));

	// list the employees sorted by ID, name and age

	std::cout << "by ID" << std::endl;
	print_out_by<id>(es);
	std::cout << std::endl;

	std::cout << "by name" << std::endl;
	print_out_by<name>(es);
	std::cout << std::endl;

	std::cout << "by age" << std::endl;
	print_out_by<age>(es);
	std::cout << std::endl;
}

/* A name record consists of the given name (e.g. "Charlie") and the family name (e.g. "Brown").
 * The full name, calculated by name_record::name() is laid out in the "phonebook order" family name + given_name.
 */

struct name_record
{
	name_record(std::string given_name_, std::string family_name_)
	: given_name(given_name_), family_name(family_name_)
	{}

	std::string name() const
	{
		std::string str = family_name;
		str += " ";
		str += given_name;
		return str;
	}

private:
	std::string given_name;
	std::string family_name;
};

std::string::size_type name_record_length(const name_record &r)
{
	return r.name().size();
}

/* multi_index_container with indices based on name_record::name() and name_record_length().
 * See Compiler specifics: Use of const_mem_fun_explicit and mem_fun_explicit for info on BOOST_MULTI_INDEX_CONST_MEM_FUN.
 */

typedef boost::multi_index::multi_index_container<
	name_record,
	boost::multi_index::indexed_by<
		boost::multi_index::ordered_unique<BOOST_MULTI_INDEX_CONST_MEM_FUN(name_record, std::string, name)>,  // member function
		boost::multi_index::ordered_non_unique<boost::multi_index::global_fun<const name_record &, std::string::size_type, name_record_length> >  // general non-member function
	>
> name_record_set;

void using_functions_as_keys()
{
	name_record_set ns;

	ns.insert(name_record("Joe", "Smith"));
	ns.insert(name_record("Robert", "Nightingale"));
	ns.insert(name_record("Robert", "Brown"));
	ns.insert(name_record("Marc", "Tuxedo"));

	// list the names in ns in phonebook order
	std::cout << "Phonenook order\n" << "---------------" << std::endl;
	for (name_record_set::iterator it = ns.begin(); it != ns.end(); ++it)
		std::cout << it->name() << std::endl;

	// list the names in ns according to their length
	std::cout << "\nLength order\n" <<  "------------" << std::endl;
	for (boost::multi_index::nth_index<name_record_set, 1>::type::iterator it1 = boost::multi_index::get<1>(ns).begin(); it1 != boost::multi_index::get<1>(ns).end(); ++it1)
		std::cout << it1->name() << std::endl;
}

}  // namespace local
}  // unnamed namespace

void multi_index()
{
	local::basic();
	local::using_functions_as_keys();
}
