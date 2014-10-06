#include <wiringPi.h>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_wiringpi {

void gpio();

}  // namespace wiringpi

int wiringpi_main(int argc, char **argv)
{
    // Initialize wiringPi library.
    wiringPiSetup();

	my_wiringpi::gpio();

	return 0;
}
