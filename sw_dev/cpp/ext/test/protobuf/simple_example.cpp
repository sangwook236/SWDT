#include "person.pb.h"
#include <fstream>
#include <iostream>


namespace {
namespace local {

void output_person()
{
	Person person;
	person.set_id(123);
	person.set_name("Bob");
	person.set_email("bob@example.com");

	std::fstream out("./protobuf_data/person.pb", std::ios::out | std::ios::binary | std::ios::trunc);
	person.SerializeToOstream(&out);
	out.close();
}

void input_person()
{
	Person person;
	std::fstream in("./protobuf_data/person.pb", std::ios::in | std::ios::binary);
	if (!person.ParseFromIstream(&in))
	{
		std::cerr << "failed to parse person.pb." << std::endl;
		return;
	}

	std::cout << "ID: " << person.id() << std::endl;
	std::cout << "name: " << person.name() << std::endl;
	if (person.has_email())
		std::cout << "e-mail: " << person.email() << std::endl;
}

}  // namespace local
}  // unnamed namespace

void simple_example()
{
	local::output_person();
	local::input_person();
}
