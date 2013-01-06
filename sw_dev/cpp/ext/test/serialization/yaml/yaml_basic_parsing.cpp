//#include "stdafx.h"
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <string>


namespace {
namespace local {

//-------------------------------------------------------------------
struct Vec3
{
	float x, y, z;
};

struct Power
{
	std::string name;
	int damage;
};

struct Monster
{
	std::string name;
	Vec3 position;
	std::vector<Power> powers;
};

// now the extraction operators for these types
void operator>>(const YAML::Node &node, Vec3 &v)
{
	node[0] >> v.x;
	node[1] >> v.y;
	node[2] >> v.z;
}

void operator>>(const YAML::Node &node, Power &power)
{
#if 1
	node["name"] >> power.name;
	node["damage"] >> power.damage;
#else
	if (const YAML::Node *name = node.FindValue("Name"))
	{
		*name >> power.name;
		std::cout << "Key 'name' exists, with value '" << power.name << "'\n";
	}
	else
	{
		std::cout << "Key 'name' doesn't exist\n";
	}
	if (const YAML::Node *damage = node.FindValue("damage"))
	{
		*damage >> power.damage;
		std::cout << "Key 'damage' exists, with value '" << power.damage << "'\n";
	}
	else
	{
		std::cout << "Key 'damage' doesn't exist\n";
	}
#endif
}

void operator>>(const YAML::Node &node, Monster &monster)
{
	node["name"] >> monster.name;
	node["position"] >> monster.position;
	const YAML::Node &powers = node["powers"];
#if 1
	for (size_t i = 0; i < powers.size(); ++i)
	{
		Power power;
		powers[i] >> power;
		monster.powers.push_back(power);
	}
#else
	for (YAML::Iterator it = powers.begin(); it != powers.end(); ++it)
	{
		// caution:
		//	YAML::Iterator's type depends on powers' type, (???)
		//	but YAML::Iterator::operator->() returns a dereferenced object.
		const YAML::NodeType::value type = it->Type();

		// run-time error:
		//	first() & second() can be called in case that Iterator's type is map, but powers's type is sequence.
/*
		const YAML::NodeType::value type2 = it.first().Type();

		std::string key;
		it.first() >> key;  // run-time error
		if (key.compare("name") == 0)
		{
			Power power;
			it.second() >> power;  // run-time error
			monster.powers.push_back(power);
		}
*/
		for (YAML::Iterator mapIt = it->begin(); mapIt != it->end(); ++mapIt)
		{
			std::string key;
			mapIt.first() >> key;
			if (key.compare("name") == 0)
			{
				std::string name;
				mapIt.second() >> name;
				std::cout << "name: " << name << std::endl;
			}
			else if (key.compare("damage") == 0)
			{
				int damage;
				mapIt.second() >> damage;
				std::cout << "damage: " << damage << std::endl;
			}
		}
	}
#endif
}

void parse_monsters()
{
	std::ifstream fin("serialization_data/yaml/monsters.yaml");
	if (!fin)
	{
		std::cout << "yaml data file not found !!!" << std::endl;
		return;
	}

	YAML::Parser parser(fin);

	YAML::Node doc;
	size_t idx = 1;
	while (parser.GetNextDocument(doc))
	{
		const size_t numSeq = doc.size();
		std::cout << "document #" << idx << ", #nodes: " << numSeq << std::endl;

#if 1
		for (size_t i = 0; i < numSeq; ++i)
		{
			Monster monster;
			doc[i] >> monster;

			std::cout << monster.name << std::endl;
		}
#else
		for (YAML::Iterator it = doc.begin(); it != doc.end(); ++it)
		{
			Monster monster;
			*it >> monster;

			std::cout << monster.name << std::endl;
		}
#endif

		++idx;
	}
}

}  // local
}  // unnamed namespace

namespace yaml {

void basic_parsing()
{
	local::parse_monsters();
}

}  // namespace yaml
