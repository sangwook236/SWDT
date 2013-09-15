//#include "stdafx.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iostream>
#include <cassert>


namespace {
namespace local {

void emit_simple_string()
{
	YAML::Emitter emitter;
	emitter << "Hello, World!";

	std::cout << "***** here's the output YAML:\n" << emitter.c_str() << std::endl;;
}

void emit_simple_sequence()
{
	YAML::Emitter emitter;
	emitter << YAML::BeginSeq;
	emitter << "eggs";
	emitter << "bread";
	emitter << "milk";
	emitter << YAML::EndSeq;

	std::cout << "***** here's the output YAML:\n" << emitter.c_str() << std::endl;;
}

void emit_simple_map()
{
	YAML::Emitter emitter;
	emitter << YAML::BeginMap;
	emitter << YAML::Key << "name";
	emitter << YAML::Value << "Ryan Braun";
	emitter << YAML::Key << "position";
	emitter << YAML::Value << "LF";
	emitter << YAML::EndMap;

	std::cout << "***** here's the output YAML:\n" << emitter.c_str() << std::endl;;
}

void emit_simple_nested_sequence_in_map()
{
	YAML::Emitter emitter;
	emitter << YAML::BeginMap;
	emitter << YAML::Key << "name";
	emitter << YAML::Value << "Barack Obama";
	emitter << YAML::Key << "children";
	emitter << YAML::Value <<
		YAML::BeginSeq << "Sasha" << "Malia" << YAML::EndSeq;
	emitter << YAML::EndMap;

	std::cout << "***** here's the output YAML:\n" << emitter.c_str() << std::endl;;
}

void comment()
{
	YAML::Emitter emitter;
	emitter << YAML::BeginMap;
	emitter << YAML::Key << "method";
	emitter << YAML::Value << "least squares";
	emitter << YAML::Comment("should we change this method?");
	emitter << YAML::EndMap;

	std::cout << "***** here's the output YAML:\n" << emitter.c_str() << std::endl;;
}

void emitter_setting()
{
	YAML::Emitter emitter;

	// the output is always UTF-8.
	// if you want to restrict the output to ASCII, use the manipulator YAML::EscapeNonAscii.
	emitter.SetOutputCharset(YAML::EscapeNonAscii);

	// if you want to permanently change a setting, there are global setters corresponding to each manipulator.
	emitter.SetIndent(4);
	emitter.SetSeqFormat(YAML::Flow);
	emitter.SetMapFormat(YAML::Flow);
}

void handle_error()
{
	YAML::Emitter emitter;
	assert(emitter.good());

	emitter << YAML::Key;
	assert(!emitter.good());
	std::cout << "***** emitter error: " << emitter.GetLastError() << std::endl;
}

void manipulator_literal()
{
	YAML::Emitter emitter;
	emitter << YAML::Literal << "A\n B\n  C";

	std::cout << "***** here's the output YAML:\n" << emitter.c_str() << std::endl;;
}

void manipulator_flow()
{
	YAML::Emitter emitter;
	emitter << YAML::Flow;
	emitter << YAML::BeginSeq << 2 << 3 << 5 << 7 << 11 << YAML::EndSeq;

	std::cout << "***** here's the output YAML:\n" << emitter.c_str() << std::endl;;
}

void manipulator_alias_and_anchor()
{
	YAML::Emitter emitter;
	emitter << YAML::BeginSeq;
	emitter << YAML::Anchor("fred");
	emitter << YAML::BeginMap;
	emitter << YAML::Key << "name" << YAML::Value << "Fred";
	emitter << YAML::Key << "age" << YAML::Value << "42";
	emitter << YAML::EndMap;
	emitter << YAML::Alias("fred");
	emitter << YAML::EndSeq;

	std::cout << "***** here's the output YAML:\n" << emitter.c_str() << std::endl;;
}

void emit_stl_container()
{
	std::vector<int> squares;
	squares.push_back(1);
	squares.push_back(4);
	squares.push_back(9);
	squares.push_back(16);

	std::map<std::string, int> ages;
	ages["Daniel"] = 26;
	ages["Jesse"] = 24;

	YAML::Emitter emitter;
	emitter << YAML::BeginSeq;
	emitter << YAML::Flow << squares;
	emitter << ages;
	emitter << YAML::EndSeq;

	std::cout << "***** here's the output YAML:\n" << emitter.c_str() << std::endl;;
}

struct Vec3
{
	int x;
	int y;
	int z;
};

YAML::Emitter & operator<<(YAML::Emitter &emitter, const Vec3 &v)
{
	emitter << YAML::Flow;
	emitter << YAML::BeginSeq << v.x << v.y << v.z << YAML::EndSeq;
	return emitter;
}

void emit_user_defined_type()
{
	Vec3 vec;
	vec.x = 1;
	vec.y = 2;
	vec.z = 3;

	YAML::Emitter emitter;
	emitter << YAML::BeginSeq;
	emitter << YAML::Flow << vec;
	emitter << YAML::EndSeq;

	std::cout << "***** here's the output YAML:\n" << emitter.c_str() << std::endl;;
}

void emit_existing_nodes()
{
	std::ifstream fin("./data/serialization/yaml/monsters.yaml");
	if (!fin)
	{
		std::cout << "yaml data file not found !!!" << std::endl;
		return;
	}

	YAML::Emitter emitter;

	YAML::Parser parser(fin);
/*
	YAML::Node doc;
	while (parser.GetNextDocument(doc))
	{
		// TODO [implement] >>
	}
*/
}

}  // local
}  // unnamed namespace

namespace my_yaml_cpp {

void basic_emitting()
{
	local::emit_simple_string();
	local::emit_simple_sequence();
	local::emit_simple_map();
	local::emit_simple_nested_sequence_in_map();

	local::comment();
	local::emitter_setting();

	local::handle_error();

	local::manipulator_literal();
	local::manipulator_flow();
	local::manipulator_alias_and_anchor();

	local::emit_stl_container();
	local::emit_user_defined_type();

	local::emit_existing_nodes();
}

}  // namespace my_yaml_cpp
