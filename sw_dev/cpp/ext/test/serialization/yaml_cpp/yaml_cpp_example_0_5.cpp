//#include "stdafx.h"
#include <yaml-cpp/yaml.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>


namespace {
namespace local {

struct Vec3
{
    Vec3()
    : x(0.0), y(0.0), z(0.0)
    {
    }
    Vec3(const double _x, const double _y, const double _z)
    : x(_x), y(_y), z(_z)
    {
    }

    double x, y, z;  // etc - make sure you have overloaded operator==
};

}  // local
}  // unnamed namespace

namespace YAML {

template<>
struct convert<local::Vec3>
{
    static Node encode(const local::Vec3 &rhs)
    {
        Node node;
        node.push_back(rhs.x);
        node.push_back(rhs.y);
        node.push_back(rhs.z);
        return node;
    }

    static bool decode(const Node &node, local::Vec3 &rhs)
    {
        if (!node.IsSequence() || 3 != node.size())
            return false;

        rhs.x = node[0].as<double>();
        rhs.y = node[1].as<double>();
        rhs.z = node[2].as<double>();
        return true;
    }
};

}

namespace {
namespace local {

void example_1()
{
    {
        YAML::Node node = YAML::Load("[1, 2, 3]");
        assert(node.Type() == YAML::NodeType::Sequence);
        assert(node.IsSequence());  // a shortcut!
    }

    // collection nodes (sequences and maps) act somewhat like STL vectors and maps
    {
        YAML::Node primes = YAML::Load("[2, 3, 5, 7, 11]");
        for (std::size_t i = 0; i < primes.size(); ++i)
            std::cout << primes[i].as<int>() << std::endl;

        for (YAML::const_iterator it = primes.begin(); it != primes.end(); ++it)
            std::cout << it->as<int>() << std::endl;

        primes.push_back(13);
        assert(primes.size() == 6);

        //
        YAML::Node lineup = YAML::Load("{1B: Prince Fielder, 2B: Rickie Weeks, LF: Ryan Braun}");
        for (YAML::const_iterator it = lineup.begin(); it != lineup.end(); ++it)
            std::cout << "Playing at " << it->first.as<std::string>() << " is " << it->second.as<std::string>() << std::endl;

#if defined(_MSC_VER)
		lineup[std::string("RF")] = "Corey Hart";
        lineup[std::string("C")] = "Jonathan Lucroy";
#else
		lineup["RF"] = "Corey Hart";
        lineup["C"] = "Jonathan Lucroy";
#endif
		assert(lineup.size() == 5);
    }

    // querying for keys does not create them automatically (this makes handling optional map entries very easy)
    {
        YAML::Node node = YAML::Load("{name: Brewers, city: Milwaukee}");
#if defined(_MSC_VER)
		if (node[std::string("name")])
            std::cout << node[std::string("name")].as<std::string>() << std::endl;
        if (node[std::string("mascot")])
            std::cout << node[std::string("mascot")].as<std::string>() << std::endl;
#else
		if (node["name"])
            std::cout << node["name"].as<std::string>() << std::endl;
        if (node["mascot"])
            std::cout << node["mascot"].as<std::string>() << std::endl;
#endif
		assert(node.size() == 2);  // the previous call didn't create a node
    }
}

void example_2()
{
    // build YAML::Node from scratch
    {
        YAML::Node node;  // starts out as null
#if defined(_MSC_VER)
        node[std::string("key")] = "value";  // it now is a map node
        node[std::string("seq")].push_back("first element");  // node["seq"] automatically becomes a sequence
        node[std::string("seq")].push_back("second element");

        node[std::string("mirror")] = node[std::string("seq")][0];  // this creates an alias
        node[std::string("seq")][0] = "1st element";  // this also changes node["mirror"]
        node[std::string("mirror")] = "element #1";  // and this changes node["seq"][0] - they're really the "same" node

        node[std::string("self")] = node;  // you can even create self-aliases
        node[node[std::string("mirror")]] = node[std::string("seq")];  // and strange loops
#else
        node["key"] = "value";  // it now is a map node
        node["seq"].push_back("first element");  // node["seq"] automatically becomes a sequence
        node["seq"].push_back("second element");

        node["mirror"] = node["seq"][0];  // this creates an alias
        node["seq"][0] = "1st element";  // this also changes node["mirror"]
        node["mirror"] = "element #1";  // and this changes node["seq"][0] - they're really the "same" node

        node["self"] = node;  // you can even create self-aliases
        node[node["mirror"]] = node["seq"];  // and strange loops
#endif

        //
        std::ofstream stream("./data/serialization/yaml/test.yaml");
        stream << node;
    }
}

void example_3()
{
    // how sequences turn i	nto maps
    {
        YAML::Node node = YAML::Load("[1, 2, 3]");
        node[1] = 5;  // still a sequence, [1, 5, 3]
        node.push_back(-3);  // still a sequence, [1, 5, 3, -3]
#if defined(_MSC_VER)
        node[std::string("key")] = "value";  // now it's a map! {0: 1, 1: 5, 2: 3, 3: -3, key: value}
#else
        node["key"] = "value";  // now it's a map! {0: 1, 1: 5, 2: 3, 3: -3, key: value}
#endif
	}

    {
        YAML::Node node = YAML::Load("[1, 2, 3]");
        node[3] = 4;  // still a sequence, [1, 2, 3, 4]
        node[10] = 10;  // now it's a map! {0: 1, 1: 2, 2: 3, 3: 4, 10: 10}
    }
}

void example_4()
{
    // converting to/from native data types
    {
        YAML::Node node = YAML::Load("{pi: 2.718, [0, 1]: integers}");

        // this needs the conversion from Node to double
#if defined(_MSC_VER)
        double pi = node[std::string("pi")].as<double>();
#else
        double pi = node["pi"].as<double>();
#endif

        // this needs the conversion from double to Node
#if defined(_MSC_VER)
        node[std::string("e")] = 2.7818;
#else
        node["e"] = 2.7818;
#endif

        // this needs the conversion from Node to std::vector<int> (*not* the other way around!)
        std::vector<int> v;
        v.push_back(0);
        v.push_back(1);
        std::string str = node[v].as<std::string>();
    }

    {
        YAML::Node node = YAML::Load("start: [1, 3, 0]");
#if defined(_MSC_VER)
        Vec3 v = node[std::string("start")].as<Vec3>();
        node[std::string("end")] = Vec3(2, -1, 0);
#else
        Vec3 v = node["start"].as<Vec3>();
        node["end"] = Vec3(2, -1, 0);
#endif
    }
}

}  // local
}  // unnamed namespace

namespace my_yaml_cpp {

void example_0_5()
{
    local::example_1();
    local::example_2();
    local::example_3();
    local::example_4();
}

}  // namespace my_yaml_cpp
