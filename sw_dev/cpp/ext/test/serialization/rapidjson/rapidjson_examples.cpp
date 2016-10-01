//#include "stdafx.h"
#include <rapidjson/reader.h>
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/filewritestream.h>
#include <rapidjson/error/en.h>
#include <string>
#include <vector>
#include <iostream>
#include <cstdio>


namespace {
namespace local {

class Person
{
public:
	Person(const std::string& name, unsigned age)
	: name_(name), age_(age)
	{}
	virtual ~Person() {}

protected:
	template <typename Writer>
	void Serialize(Writer &writer) const
	{
		// This base class just write out name-value pairs, without wrapping within an object.
		writer.String("name");
		writer.String(name_.c_str(), (rapidjson::SizeType)name_.length());	 // Suppling length of string is faster.

		writer.String("age");
		writer.Uint(age_);
	}

private:
	std::string name_;
	unsigned age_;
};

class Education
{
public:
	Education(const std::string &school, double GPA)
	: school_(school), GPA_(GPA)
	{}

	template <typename Writer>
	void Serialize(Writer &writer) const
	{
		writer.StartObject();

		writer.String("school");
		writer.String(school_.c_str(), (rapidjson::SizeType)school_.length());

		writer.String("GPA");
		writer.Double(GPA_);

		writer.EndObject();
	}

private:
	std::string school_;
	double GPA_;
};

class Dependent : public Person
{
public:
	Dependent(const std::string &name, unsigned age, Education *education = 0)
	: Person(name, age), education_(education)
	{}
	Dependent(const Dependent &rhs)
	: Person(rhs)
	{
		education_ = (0 == rhs.education_) ? 0 : new Education(*rhs.education_);
	}
	~Dependent() { delete education_; }

	template <typename Writer>
	void Serialize(Writer &writer) const
	{
		writer.StartObject();

		Person::Serialize(writer);

		writer.String("education");
		if (education_)
			education_->Serialize(writer);
		else
			writer.Null();

		writer.EndObject();
	}

private:
	Education *education_;
};

class Employee : public Person
{
public:
	Employee(const std::string &name, unsigned age, bool married)
	: Person(name, age), married_(married)
	{}

	void AddDependent(const Dependent &dependent)
	{
		dependents_.push_back(dependent);
	}

	template <typename Writer>
	void Serialize(Writer &writer) const
	{
		writer.StartObject();

		Person::Serialize(writer);

		writer.String("married");
		writer.Bool(married_);

		writer.String(("dependents"));
		writer.StartArray();
		for (std::vector<Dependent>::const_iterator dependentItr = dependents_.begin(); dependentItr != dependents_.end(); ++dependentItr)
			dependentItr->Serialize(writer);
		writer.EndArray();

		writer.EndObject();
	}

private:
	bool married_;
	std::vector<Dependent> dependents_;
};

}  // namespace local
}  // unnamed namespace

namespace my_rapidjson {

// REF [file] >> ${RAPIDJSON_HOME}/example/serialize/serialize.cpp.
void serialize_example()
{
	std::vector<local::Employee> employees;

	employees.push_back(local::Employee("Milo YIP", 34, true));
	employees.back().AddDependent(local::Dependent("Lua YIP", 3, new local::Education("Happy Kindergarten", 3.5)));
	employees.back().AddDependent(local::Dependent("Mio YIP", 1));

	employees.push_back(local::Employee("Percy TSE", 30, false));

	rapidjson::StringBuffer sb;
	rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(sb);

	writer.StartArray();
	for (std::vector<local::Employee>::const_iterator employeeItr = employees.begin(); employeeItr != employees.end(); ++employeeItr)
		employeeItr->Serialize(writer);
	writer.EndArray();
}

// REF [file] >> ${RAPIDJSON_HOME}/example/condense/condense.cpp.
// REF [file] >> ${RAPIDJSON_HOME}/example/pretty/pretty.cpp.
void condense_and_pretty_example()
{
	// Prepare JSON reader and input stream.
	rapidjson::Reader reader;
	char readBuffer[65536];
#if 0
	rapidjson::FileReadStream is(stdin, readBuffer, sizeof(readBuffer));
#else
	//FILE *fp = fopen("./data/serialization/json/glossary.json", "r+");
	FILE *fp = fopen("./data/serialization/json/menu.json", "r+");
	//FILE *fp = fopen("./data/serialization/json/sample.json", "r+");
	//FILE *fp = fopen("./data/serialization/json/webapp.json", "r+");
	//FILE *fp = fopen("./data/serialization/json/widget.json", "r+");
	rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
#endif

	// Prepare JSON writer and output stream.
	char writeBuffer[65536];
	rapidjson::FileWriteStream os(stdout, writeBuffer, sizeof(writeBuffer));
#if 0
	rapidjson::Writer<rapidjson::FileWriteStream> writer(os);
#else
	rapidjson::PrettyWriter<rapidjson::FileWriteStream> writer(os);
#endif

	// JSON reader parse from the input stream and let writer generate the output.
	if (!reader.Parse<0>(is, writer))
	{
		std::cerr << "\nError(" << (unsigned)reader.GetErrorOffset() << "): " << GetParseError_En(reader.GetParseErrorCode()) << std::endl;
		return;
	}

	fclose(fp);
}

}  // namespace my_rapidjson
