#include <nmea/nmea.h>
#include <iostream>
#include <string>
#include <list>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

namespace my_nmea {

// [ref] ${NMEALIB_HOME)/samples/math/main.c.
void math()
{
	const int NUM_POINTS = 4;

	std::list<std::string> bufs;
    bufs.push_back("$GPRMC,213916.199,A,4221.0377,N,07102.9778,W,0.00,,010207,,,A*6A\r\n");
    bufs.push_back("$GPRMC,213917.199,A,4221.0510,N,07102.9549,W,0.23,175.43,010207,,,A*77\r\n");
    bufs.push_back("$GPRMC,213925.000,A,4221.1129,N,07102.9146,W,0.00,,010207,,,A*68\r\n");
    bufs.push_back("$GPRMC,111609.14,A,5001.27,N,3613.06,E,11.2,0.0,261206,0.0,E*50\r\n");

    nmeaPARSER parser;
    nmea_parser_init(&parser);

    nmeaPOS pos[NUM_POINTS];
	int idx = 0;
	for (std::list<std::string>::const_iterator cit = bufs.begin(); cit != bufs.end(); ++cit)
    {
        nmeaINFO info;
        nmea_zero_INFO(&info);
        const int result = nmea_parse(&parser, cit->c_str(), (int)cit->length(), &info);        
        nmea_info2pos(&info, &pos[idx++]);
    }

    nmea_parser_destroy(&parser);

	//
    double dist[NUM_POINTS][2], azimuth[NUM_POINTS][2]; 
    for (int i = 0; i < NUM_POINTS; ++i)
    {
        dist[i][0] = nmea_distance(&pos[0], &pos[i]);
        dist[i][1] = nmea_distance_ellipsoid(&pos[0], &pos[i], &azimuth[i][0], &azimuth[i][1]);
    }

    nmeaPOS pos_moved[NUM_POINTS][2];
	double azimuth_moved[NUM_POINTS];
    int result[NUM_POINTS][2];
    for (int i = 0; i < NUM_POINTS; ++i)
    {
        result[i][0] = nmea_move_horz(&pos[0], &pos_moved[i][0], azimuth[i][0], dist[i][0]);
        result[i][1] = nmea_move_horz_ellipsoid(&pos[0], &pos_moved[i][1], azimuth[i][0], dist[i][0], &azimuth_moved[i]);
    }

    // output of results.
    std::cout << "coordinate points:" << std::endl;
    for (int i = 0; i < NUM_POINTS; ++i)
    {
        std::cout << "P" << i << " in radians: lat:" << pos[i].lat << " lon:" << pos[i].lon << "  \tin degree: lat:" << nmea_radian2degree(pos[i].lat) << "?lon:" << nmea_radian2degree(pos[i].lon) << std::endl;
    }

    std::cout << std::endl << "calculation results:" << std::endl;
    for (int i = 0; i < NUM_POINTS; ++i)
    {
        std::cout << std::endl;
        std::cout << "Distance P0 to P" << i << "\ton spheroid:  " << dist[i][0] << " m" << std::endl;
        std::cout << "Distance P0 to P" << i << "\ton ellipsoid: " << dist[i][1] << " m" << std::endl;
        std::cout << "Azimuth  P0 to P" << i << "\tat start: " << nmea_radian2degree(azimuth[i][0]) << "\tat end: " << nmea_radian2degree(azimuth[i][1]) << std::endl;
        std::cout << "Move     P0 to P" << i << "\t         \tAzimuth at end: " << nmea_radian2degree(azimuth_moved[i]) << std::endl;
        std::cout << "Move     P0 to P" << i << "\ton spheroid:  " << (result[i][0] == 1 ? "OK" : "nOK") << " lat:" << nmea_radian2degree(pos_moved[i][0].lat) << "?lon:" << nmea_radian2degree(pos_moved[i][0].lon) << std::endl;
        std::cout << "Move     P0 to P" << i << "\ton ellipsoid: " << (result[i][1] == 1 ? "OK" : "nOK") << " lat:" << nmea_radian2degree(pos_moved[i][1].lat) << "?lon:" << nmea_radian2degree(pos_moved[i][1].lon) << std::endl;
        std::cout << "Move     P0 to P" << i << "\toriginal:        lat:" << nmea_radian2degree(pos[i].lat) << "?lon:" << nmea_radian2degree(pos[i].lon) << std::endl;
    }
}

}  // namespace my_nmea
