#include "protobuf_person.pb.h"
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

	std::fstream out("./data/serialization/protobuf/person.pb", std::ios::out | std::ios::binary | std::ios::trunc);
	person.SerializeToOstream(&out);
	out.close();
}

void input_person()
{
	Person person;
	std::fstream in("./data/serialization/protobuf/person.pb", std::ios::in | std::ios::binary);
	if (!person.ParseFromIstream(&in))
	{
		std::cerr << "Failed to parse person.pb." << std::endl;
		return;
	}

	std::cout << "ID: " << person.id() << std::endl;
	std::cout << "name: " << person.name() << std::endl;
	if (person.has_email())
		std::cout << "e-mail: " << person.email() << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_protobuf {

void proto2_example()
{
	local::output_person();
	local::input_person();
}

}  // namespace my_protobuf
