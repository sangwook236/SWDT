//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_sqlite {

void basic();
void encryption_decryption();

}  // namespace my_sqlite

int sqlite_main(int argc, char *argv[])
{
	
	my_sqlite::basic();
	//my_sqlite::encryption_decryption();

	return 0;
}