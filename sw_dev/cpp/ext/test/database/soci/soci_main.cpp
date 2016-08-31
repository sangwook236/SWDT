//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_soci {

void sqlite_basic();
void postgresql_basic();

}  // namespace my_soci

int soci_main(int argc, char *argv[])
{
	my_soci::sqlite_basic();
	//my_soci::postgresql_basic();

	return 0;
}