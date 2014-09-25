#include <nmea/nmea.h>
#include <iostream>
#ifdef NMEA_WIN
#include <windows.h>
#else
#include <unistd.h>
#endif


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_nmea {

// [ref] ${NMEALIB_HOME)/samples/generate/main.c.
void generate()
{
	nmeaINFO info;
	nmea_zero_INFO(&info);
	info.sig = 3;
	info.fix = 3;
	info.lat = 5000.0;
	info.lon = 3600.0;
	info.speed = 2.14 * NMEA_TUS_MS;
	info.elv = 10.86;
	info.satinfo.inuse = 1;
	info.satinfo.inview = 1;
	/*
	info.satinfo.sat[0].id = 1;
	info.satinfo.sat[0].in_use = 1;
	info.satinfo.sat[0].elv = 50;
	info.satinfo.sat[0].azimuth = 0;
	info.satinfo.sat[0].sig = 99;
	*/

	char buff[2048];
	for (int i = 0; i < 10; ++i)
	{
		const int gen_sz = nmea_generate(buff, 2048, &info, GPGGA | GPGSA | GPGSV | GPRMC | GPVTG);

		// for display.
		buff[gen_sz] = '\0';
		std::cout << buff << std::endl;

#ifdef NMEA_WIN
        Sleep(500);
#else
        usleep(500000);
#endif        

		info.speed += .1;
	}
}

// [ref] ${NMEALIB_HOME)/samples/generator/main.c.
void use_generator()
{

	nmeaINFO info;
	nmea_zero_INFO(&info);

	nmeaGENERATOR *gen = nmea_create_generator(NMEA_GEN_ROTATE, &info);
	if (0 == gen)
	{
		std::cerr << "generator not created" << std::endl;
		return;
	}

	char buff[2048];
	for (int i = 0; i < 10000; ++i)
	{
		const int gen_sz = nmea_generate_from(buff, 2048, &info, gen, GPGGA | GPGSA | GPGSV | GPRMC | GPVTG);

		// for display.
		buff[gen_sz] = '\0';
		std::cout << buff << std::endl;

#ifdef NMEA_WIN
		Sleep(500);
#else
		usleep(500000);        
#endif
	}

	nmea_gen_destroy(gen);
}

}  // namespace my_nmea
