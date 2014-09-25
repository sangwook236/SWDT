#include <nmea/nmea.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_nmea {

void parse_string();
void parse_file();
void math();
void generate();
void use_generator();

}  // namespace my_nmea

int nmea_main(int argc, char *argv[])
{
	//my_nmea::parse_string();
	//my_nmea::parse_file();

	//my_nmea::generate();
	//my_nmea::use_generator();

	my_nmea::math();

	return 0;
}
