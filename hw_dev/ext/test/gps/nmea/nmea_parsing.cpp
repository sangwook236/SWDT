#include <nmea/nmea.h>
#include <fstream>
#include <iostream>
#include <string>
#include <list>


namespace {
namespace local {

void handle_trace(const char *str, int str_size)
{
	std::cerr << "Trace: " << str << std::endl;
}

void handle_error(const char *str, int str_size)
{
	std::cerr << "Error: " << str << std::endl;
}

}  // namespace local
}  // unnamed namespace

namespace my_nmea {

// [ref] ${NMEALIB_HOME)/samples/parse/main.c.
void parse_string()
{
	std::list<std::string> bufs;
	bufs.push_back("$GPRMC,173843,A,3349.896,N,11808.521,W,000.0,360.0,230108,013.4,E*69\r\n");
	bufs.push_back("$GPGGA,111609.14,5001.27,N,3613.06,E,3,08,0.0,10.2,M,0.0,M,0.0,0000*70\r\n");
	bufs.push_back("$GPGSV,2,1,08,01,05,005,80,02,05,050,80,03,05,095,80,04,05,140,80*7f\r\n");
	bufs.push_back("$GPGSV,2,2,08,05,05,185,80,06,05,230,80,07,05,275,80,08,05,320,80*71\r\n");
	bufs.push_back("$GPGSA,A,3,01,02,03,04,05,06,07,08,00,00,00,00,0.0,0.0,0.0*3a\r\n");
	bufs.push_back("$GPRMC,111609.14,A,5001.27,N,3613.06,E,11.2,0.0,261206,0.0,E*50\r\n");
	bufs.push_back("$GPVTG,217.5,T,208.8,M,000.00,N,000.01,K*4C\r\n");

	nmeaINFO info;
	nmea_zero_INFO(&info);

	nmeaPARSER parser;
	nmea_parser_init(&parser);

	nmeaPOS dpos;
	std::size_t idx = 0;
	for (std::list<std::string>::const_iterator cit = bufs.begin(); cit != bufs.end(); ++cit)
	{
		nmea_parse(&parser, cit->c_str(), (int)cit->length(), &info);
		nmea_info2pos(&info, &dpos);

		std::cout << ++idx << ", Lat: " << dpos.lat << ", Lon: " << dpos.lon << ", Sig: " << info.sig << ", Fix: " << info.fix << std::endl;
	}

	nmea_parser_destroy(&parser);
}

// [ref] ${NMEALIB_HOME)/samples/parse_file/main.c.
void parse_file()
{
	std::ifstream fin("./data/gps/gpslog.txt");
	if (!fin.is_open())
	{
		std::cerr << "input file not found" << std::endl;
		return;
	}

	nmea_property()->trace_func = &local::handle_trace;
	nmea_property()->error_func = &local::handle_error;

	nmeaINFO info;
	nmea_zero_INFO(&info);

	nmeaPARSER parser;
	nmea_parser_init(&parser);

	nmeaPOS dpos;
	char buf[2048];
	std::size_t idx = 0;
	while (!fin.eof())
	{
		fin.read(buf, 100);
		const int size = (int)fin.gcount();

		// for checking.
		//buf[size] = '\0';
		//std::cout << buf;

		nmea_parse(&parser, buf, size, &info);
		nmea_info2pos(&info, &dpos);

		std::cout << ++idx << ", Lat: " << dpos.lat << ", Lon: " << dpos.lon << ", Sig: " << info.sig << ", Fix: " << info.fix << std::endl;
	}
	fin.close();

	nmea_parser_destroy(&parser);
}

}  // namespace my_nmea
