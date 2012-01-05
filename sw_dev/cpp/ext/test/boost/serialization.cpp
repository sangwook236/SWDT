#include "stdafx.h"
#include <boost/archive/tmpdir.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/utility.hpp>

#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
//#include <cstddef>  // NULL


// NVP: a name-value pair for XML serialization
#define __MACRO_USE_NVP_FOR_XML_SERIALIZATION_ 1

//namespace {
//namespace local {

/////////////////////////////////////////////////////////////
// gps coordinate
//
// llustrates serialization for a simple type
//
class gps_position
{
	friend std::ostream & operator<<(std::ostream &os, const gps_position &gp);
	friend class ::boost::serialization::access;

	int degrees;
	int minutes;
	float seconds;

	template<class Archive>
	void serialize(Archive & ar, const unsigned int /* file_version */)
	{
#if !defined(__MACRO_USE_NVP_FOR_XML_SERIALIZATION_)
		ar & degrees & minutes & seconds;
#else
		ar & BOOST_SERIALIZATION_NVP(degrees)
			& BOOST_SERIALIZATION_NVP(minutes)
			& BOOST_SERIALIZATION_NVP(seconds);
#endif
	}

public:
	// every serializable class needs a constructor
	gps_position()
	{}
	gps_position(int d, int m, float s)
	: degrees(d), minutes(m), seconds(s)
	{}
};

std::ostream & operator<<(std::ostream &os, const gps_position &gp)
{
	return os << ' ' << gp.degrees << (unsigned char)186 << gp.minutes << '\'' << gp.seconds << '"';
}

/////////////////////////////////////////////////////////////
// One bus stop
//
// illustrates serialization of serializable members
//

class bus_stop
{
	friend class boost::serialization::access;
	friend std::ostream & operator<<(std::ostream &os, const bus_stop &gp);

	virtual std::string description() const = 0;

	gps_position latitude;
	gps_position longitude;

	template<class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
#if !defined(__MACRO_USE_NVP_FOR_XML_SERIALIZATION_)
		ar & latitude;
		ar & longitude;
#else
		ar & BOOST_SERIALIZATION_NVP(latitude);
		ar & BOOST_SERIALIZATION_NVP(longitude);
#endif
	}

protected:
	bus_stop(const gps_position &lat, const gps_position &lon)
	: latitude(lat), longitude(lon)
	 {}

public:
	bus_stop()
	{}
	virtual ~bus_stop()
	{}
};
BOOST_SERIALIZATION_ASSUME_ABSTRACT(bus_stop)

std::ostream & operator<<(std::ostream &os, const bus_stop &bs)
{
	return os << bs.latitude << bs.longitude << ' ' << bs.description();
}

/////////////////////////////////////////////////////////////
// Several kinds of bus stops
//
// illustrates serialization of derived types
//
class bus_stop_corner : public bus_stop
{
	friend class boost::serialization::access;

	std::string street1;
	std::string street2;

	virtual std::string description() const
	{
		return street1 + " and " + street2;
	}

	template<class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		// save/load base class information
#if !defined(__MACRO_USE_NVP_FOR_XML_SERIALIZATION_)
		ar & boost::serialization::base_object<bus_stop>(*this);
		ar & street1 & street2;
#else
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(bus_stop);
		ar & BOOST_SERIALIZATION_NVP(street1);
		ar & BOOST_SERIALIZATION_NVP(street2);
#endif
	}

public:
	bus_stop_corner()
	{}
	bus_stop_corner(const gps_position &lat, const gps_position &lon, const std::string &s1, const std::string &s2)
	: bus_stop(lat, lon), street1(s1), street2(s2)
	{}
};

class bus_stop_destination : public bus_stop
{
	friend class boost::serialization::access;

	std::string name;

	virtual std::string description() const
	{
		return name;
	}

	template<class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
#if !defined(__MACRO_USE_NVP_FOR_XML_SERIALIZATION_)
		ar & boost::serialization::base_object<bus_stop>(*this) & name;
#else
		ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(bus_stop)
			& BOOST_SERIALIZATION_NVP(name);
#endif
	}

public:

	bus_stop_destination()
	{}
	bus_stop_destination(const gps_position &lat, const gps_position &lon, const std::string &n)
	: bus_stop(lat, lon), name(n)
	{}
};

/////////////////////////////////////////////////////////////
// a bus route is a collection of bus stops
//
// illustrates serialization of STL collection templates.
//
// illustrates serialzation of polymorphic pointer (bus stop *);
//
// illustrates storage and recovery of shared pointers is correct
// and efficient.  That is objects pointed to by more than one
// pointer are stored only once.  In such cases only one such
// object is restored and pointers are restored to point to it
//
class bus_route
{
	friend class boost::serialization::access;
	friend std::ostream & operator<<(std::ostream &os, const bus_route &br);

	typedef bus_stop * bus_stop_pointer;
	std::list<bus_stop_pointer> stops;
	
	template<class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
		// in this program, these classes are never serialized directly but rather
		// through a pointer to the base class bus_stop. So we need a way to be
		// sure that the archive contains information about these derived classes.
		//ar.template register_type<bus_stop_corner>();
		ar.register_type(static_cast<bus_stop_corner *>(NULL));
		//ar.template register_type<bus_stop_destination>();
		ar.register_type(static_cast<bus_stop_destination *>(NULL));
		// serialization of stl collections is already defined
		// in the header
#if !defined(__MACRO_USE_NVP_FOR_XML_SERIALIZATION_)
		ar & stops;
#else
		ar & BOOST_SERIALIZATION_NVP(stops);
#endif
	}

public:
	bus_route()
	{}
	
	void append(bus_stop *_bs)
	{
		stops.insert(stops.end(), _bs);
	}
};
BOOST_CLASS_VERSION(bus_route, 1)

std::ostream & operator<<(std::ostream &os, const bus_route &br)
{
	// note: we're displaying the pointer to permit verification
	// that duplicated pointers are properly restored.
	for (std::list<bus_stop *>::const_iterator it = br.stops.begin(); it != br.stops.end(); ++it)
	{
		os << '\n' << std::hex << "0x" << *it << std::dec << ' ' << **it;
	}

	return os;
}

/////////////////////////////////////////////////////////////
// a bus schedule is a collection of routes each with a starting time
//
// Illustrates serialization of STL objects(pair) in a non-intrusive way.
// See definition of operator<< <pair<F, S> >(ar, pair) and others in
// serialization.hpp
// 
// illustrates nesting of serializable classes
//
// illustrates use of version number to automatically grandfather older
// versions of the same class.

class bus_schedule
{
public:
	// note: this structure was made public. because the friend declarations
	// didn't seem to work as expected.
	struct trip_info
	{
		template<class Archive>
		void serialize(Archive &ar, const unsigned int file_version)
		{
#if !defined(__MACRO_USE_NVP_FOR_XML_SERIALIZATION_)
			// in versions 2 or later
			if (file_version >= 2)
				// read the drivers name
				ar & driver;
			// all versions have the follwing info
			ar & hour & minute;
#else
			// in versions 2 or later
			if(file_version >= 2)
				// read the drivers name
				ar & BOOST_SERIALIZATION_NVP(driver);
			// all versions have the follwing info
			ar  & BOOST_SERIALIZATION_NVP(hour)
				& BOOST_SERIALIZATION_NVP(minute);
#endif
		}

		// starting time
		int hour;
		int minute;
		// only after system shipped was the driver's name added to the class
		std::string driver;

		trip_info()
		{}
		trip_info(int h, int m, const std::string &d)
		: hour(h), minute(m), driver(d)
		{}
	};

private:
	friend class boost::serialization::access;
	friend std::ostream & operator<<(std::ostream &os, const bus_schedule &bs);
	friend std::ostream & operator<<(std::ostream &os, const bus_schedule::trip_info &ti);

	std::list<std::pair<trip_info, bus_route *> > schedule;

	template<class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
#if !defined(__MACRO_USE_NVP_FOR_XML_SERIALIZATION_)
		ar & schedule;
#else
		ar & BOOST_SERIALIZATION_NVP(schedule);
#endif
	}

public:
	void append(const std::string &d, int h, int m, bus_route *br)
	{
		schedule.insert(schedule.end(), std::make_pair(trip_info(h, m, d), br));
	}

	bus_schedule()
	{}
};
BOOST_CLASS_VERSION(bus_schedule::trip_info, 3)
BOOST_CLASS_VERSION(bus_schedule, 2)

std::ostream & operator<<(std::ostream &os, const bus_schedule::trip_info &ti)
{
	return os << '\n' << ti.hour << ':' << ti.minute << ' ' << ti.driver << ' ';
}

std::ostream & operator<<(std::ostream &os, const bus_schedule &bs)
{
	for (std::list<std::pair<bus_schedule::trip_info, bus_route *> >::const_iterator it = bs.schedule.begin(); it != bs.schedule.end(); ++it)
	{
		os << it->first << *(it->second);
	}
	return os;
}

void save_schedule(const bus_schedule &s, const char *filename)
{
	// make an archive
	std::ofstream ofs(filename);
    assert(ofs.good());
#if !defined(__MACRO_USE_NVP_FOR_XML_SERIALIZATION_)
	boost::archive::text_oarchive oa(ofs);
	oa << s;
#else
    boost::archive::xml_oarchive oa(ofs);
	oa << BOOST_SERIALIZATION_NVP(s);
#endif
}

void restore_schedule(bus_schedule &s, const char *filename)
{
	// open the archive
	std::ifstream ifs(filename);
    assert(ifs.good());
#if !defined(__MACRO_USE_NVP_FOR_XML_SERIALIZATION_)
	boost::archive::text_iarchive ia(ifs);
	// restore the schedule from the archive
	ia >> s;
#else
	boost::archive::xml_iarchive ia(ifs);
	// restore the schedule from the archive
	ia >> BOOST_SERIALIZATION_NVP(s);
#endif
}

void serialization_bus_schedule()
{   
	// make the schedule
	bus_schedule original_schedule;

	// fill in the data
	// make a few stops
	bus_stop *bs0 = new bus_stop_corner(
		gps_position(34, 135, 52.560f),
		gps_position(134, 22, 78.30f),
		"24th Street", "10th Avenue"
	);
	bus_stop *bs1 = new bus_stop_corner(
		gps_position(35, 137, 23.456f),
		gps_position(133, 35, 54.12f),
		"State street", "Cathedral Vista Lane"
	);
	bus_stop *bs2 = new bus_stop_destination(
		gps_position(35, 136, 15.456f),
		gps_position(133, 32, 15.300f),
		"White House"
	);
	bus_stop *bs3 = new bus_stop_destination(
		gps_position(35, 134, 48.789f),
		gps_position(133, 32, 16.230f),
		"Lincoln Memorial"
	);

	// make a routes
	bus_route route0;
	route0.append(bs0);
	route0.append(bs1);
	route0.append(bs2);

	// add trips to schedule
	original_schedule.append("bob", 6, 24, &route0);
	original_schedule.append("bob", 9, 57, &route0);
	original_schedule.append("alice", 11, 02, &route0);

	// make aother routes
	bus_route route1;
	route1.append(bs3);
	route1.append(bs2);
	route1.append(bs1);

	// add trips to schedule
	original_schedule.append("ted", 7, 17, &route1);
	original_schedule.append("ted", 9, 38, &route1);
	original_schedule.append("alice", 11, 47, &route1);

	// display the complete schedule
	std::cout << "original schedule";
	std::cout << original_schedule;

	{
#if !defined(__MACRO_USE_NVP_FOR_XML_SERIALIZATION_)
		const std::string filename(std::string(boost::archive::tmpdir()) + std::string("/demofile.txt"));
#else
		const std::string filename(std::string(boost::archive::tmpdir()) + std::string("/demofile.xml"));
#endif

		// save the schedule
		save_schedule(original_schedule, filename.c_str());

		// ... some time later
		// make  a new schedule
		bus_schedule new_schedule;

		restore_schedule(new_schedule, filename.c_str());

		// and display
		std::cout << "\nrestored schedule";
		std::cout << new_schedule;
		// should be the same as the old one. (except for the pointer values)
	}

	delete bs0;
	delete bs1;
	delete bs2;
	delete bs3;
}

class gps_position_intrusive
{
private:
	friend class boost::serialization::access;

	// When the class Archive corresponds to an output archive, the
	// & operator is defined similar to <<.  Likewise, when the class Archive
	// is a type of input archive the & operator is defined similar to >>.
	template<class Archive>
	void serialize(Archive &ar, const unsigned int version)
	{
#if !defined(__MACRO_USE_NVP_FOR_XML_SERIALIZATION_)
		ar & degrees
			& minutes
			& seconds;
#else
		ar & BOOST_SERIALIZATION_NVP(degrees)
			& BOOST_SERIALIZATION_NVP(minutes)
			& BOOST_SERIALIZATION_NVP(seconds);
#endif
	}

	int degrees;
	int minutes;
	float seconds;

public:
	gps_position_intrusive()
	{}
	gps_position_intrusive(int d, int m, float s)
	: degrees(d), minutes(m), seconds(s)
	{}
};

void serialization_intrusive()
{
#if !defined(__MACRO_USE_NVP_FOR_XML_SERIALIZATION_)
	const std::string filename("boost_data/gps_intrusive_archive.txt");
#else
	const std::string filename("boost_data/gps_intrusive_archive.xml");
#endif

	// save data to archive
	{
		// create class instance
		const gps_position_intrusive g(35, 59, 24.567f);

		// create and open a character archive for output
		std::ofstream ofs(filename.c_str());
#if !defined(__MACRO_USE_NVP_FOR_XML_SERIALIZATION_)
		boost::archive::text_oarchive oa(ofs);
		// write class instance to archive
		oa << g;
#else
		boost::archive::xml_oarchive oa(ofs);
		oa << BOOST_SERIALIZATION_NVP(g);
#endif
		// archive and stream closed when destructors are called
	}

	// ... some time later restore the class instance to its orginal state
	{
		gps_position_intrusive newg;

		// create and open an archive for input
		std::ifstream ifs(filename.c_str());
#if !defined(__MACRO_USE_NVP_FOR_XML_SERIALIZATION_)
		boost::archive::text_iarchive ia(ifs);
		// read class state from archive
		ia >> newg;
#else
		boost::archive::xml_iarchive ia(ifs);
		ia >> BOOST_SERIALIZATION_NVP(newg);
#endif
		// archive and stream closed when destructors are called
	}
}

class gps_position_nonintrusive
{
public:
	int degrees;
	int minutes;
	float seconds;

	gps_position_nonintrusive()
	{}
	gps_position_nonintrusive(int d, int m, float s)
	: degrees(d), minutes(m), seconds(s)
	{}
};

namespace boost {
namespace serialization {

template<class Archive>
void serialize(Archive &ar, gps_position_nonintrusive &g, const unsigned int version)
{
#if !defined(__MACRO_USE_NVP_FOR_XML_SERIALIZATION_)
	ar & g.degrees
		& g.minutes
		& g.seconds;
#else
	ar & BOOST_SERIALIZATION_NVP(g.degrees)
		& BOOST_SERIALIZATION_NVP(g.minutes)
		& BOOST_SERIALIZATION_NVP(g.seconds);
#endif
}

} // namespace serialization
} // namespace boost

void serialization_nonintrusive()
{
#if !defined(__MACRO_USE_NVP_FOR_XML_SERIALIZATION_)
	const std::string filename("boost_data/gps_nonintrusive_archive.txt");
#else
	const std::string filename("boost_data/gps_nonintrusive_archive.xml");
#endif

	// save data to archive
	{
		// create class instance
		const gps_position_nonintrusive g(35, 59, 24.567f);

		// create and open a character archive for output
		std::ofstream ofs(filename.c_str());
#if !defined(__MACRO_USE_NVP_FOR_XML_SERIALIZATION_)
		boost::archive::text_oarchive oa(ofs);
		// write class instance to archive
		oa << g;
#else
		boost::archive::xml_oarchive oa(ofs);
		oa << BOOST_SERIALIZATION_NVP(g);
#endif
		// archive and stream closed when destructors are called
	}

	// ... some time later restore the class instance to its orginal state
	{
		gps_position_nonintrusive newg;

		// create and open an archive for input
		std::ifstream ifs(filename.c_str());
#if !defined(__MACRO_USE_NVP_FOR_XML_SERIALIZATION_)
		boost::archive::text_iarchive ia(ifs);
		// read class state from archive
		ia >> newg;
#else
		boost::archive::xml_iarchive ia(ifs);
		ia >> BOOST_SERIALIZATION_NVP(newg);
#endif
		// archive and stream closed when destructors are called
	}
}

//}  // namespace local
//}  // unnamed namespace

void serialization()
{
	serialization_intrusive();
	serialization_nonintrusive();

	serialization_bus_schedule();
}
