//#include "stdafx.h"
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace sqlite {

void basic();
void encryption_decryption();

}  // namespace sqlite

int sqlite_main(int argc, char *argv[])
{
	
	sqlite::basic();
	//sqlite::encryption_decryption();

	return 0;
}