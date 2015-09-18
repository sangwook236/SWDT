//#include "stdafx.h"
#include "../arff_lib/arff_data.h"
#include "../arff_lib/arff_parser.h"
#include <iostream>
#include <memory>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_arff {

}  // namespace my_arff

int arff_main(int argc, char *argv[])
{
	//ArffParser parser("./data/file_format/case1.arff");
	//ArffParser parser("./data/file_format/case2.arff");
	//ArffParser parser("./data/file_format/case3.arff");
	ArffParser parser("./data/file_format/case4.arff");

	ArffData *data = parser.parse();

    // relation => @RELATION.
	const std::string relationName = data->get_relation_name();
	std::cout << "relation name = " << relationName << std::endl;

    // attributes => @ATTRIBUTE.
    {
        const int32 numAttr = data->num_attributes();
        std::cout << "the number of attributes = " << numAttr << std::endl;

        for (int32 i = 0; i < numAttr; ++i)
        {
            const ArffAttr *attr = data->get_attr(i);
            const std::string name = attr->name();
            const ArffValueEnum type = attr->type();

            std::cout << "\tattribute #" << i << " : name = " << name << ", type = " << arff_value2str(type) << std::endl;

            if (NOMINAL == type)
            {
                const ArffNominal nominal = data->get_nominal(name);
                std::cout << "\t\t";
                for (ArffNominal::const_iterator cit = nominal.begin(); cit != nominal.end(); ++cit)
                    std::cout << *cit << ", ";
                std::cout << std::endl;
            }
        }
    }

    // instance => @DATA.
    {
        const int32 numInst = data->num_instances();
        std::cout << "the number of instances = " << numInst << std::endl;

        for (int32 i = 0; i < numInst; ++i)
        {
            const ArffInstance *inst = data->get_instance(i);

            std::cout << '\t';
            for (int32 k = 0; k < inst->size(); ++k)
            {
                const ArffValue *value = inst->get(k);

                if (value->missing())
                    std::cout << '?';
                else
                {
                    switch (value->type())
                    {
                    case INTEGER:
                        std::cout << int32(*value);
                        break;
                    case FLOAT:
                        std::cout << float(*value);
                        break;
                    case DATE:
                        std::cout << std::string(*value);  // FIXME [check] >> is it right?
                        break;
                    case STRING:
                        std::cout << std::string(*value);
                       break;
                    case NUMERIC:
                        std::cout << std::string(*value);  // FIXME [check] >> is it right?
                        break;
                    case NOMINAL:
                        std::cout << std::string(*value);  // FIXME [check] >> is it right?
                        break;
                    case UNKNOWN_VAL:
                    default:
                        std::cout << '#';
                        break;
                    }
                }
                std::cout << ',';
            }
            std::cout << std::endl;
        }
    }

    //delete data;

	return 0;
}
