#include <boost/date_time/posix_time/posix_time.hpp>
#include <iostream>


void gregorian_basic();
void posix_time_basic();

void date_time()
{
	gregorian_basic();
	posix_time_basic();
}

void gregorian_basic()
{
	boost::gregorian::date weekstart(2002, boost::date_time::Feb, 1);
	boost::gregorian::date weekend = weekstart + boost::gregorian::weeks(1);

	boost::gregorian::date d1(2008, boost::date_time::Nov, 10);
	boost::gregorian::date d2 = d1 + boost::gregorian::days(5);

	boost::gregorian::date today = boost::gregorian::day_clock::local_day();
	//boost::gregorian::date::ymd_type today = boost::gregorian::day_clock::local_day_ymd();
	//boost::gregorian::date today = boost::gregorian::day_clock::universal_day();
	//boost::gregorian::date::ymd_type today = boost::gregorian::day_clock::universal_day_ymd();
	if (d2 >= today) {}  // date comparison operators

	boost::gregorian::date_period thisWeek(d1, d2);
	if (thisWeek.contains(today)) {}  // do something
	
	// iterate and print the week
	boost::gregorian::day_iterator itr(weekstart);
	while (itr <= weekend) {
		std::cout << (*itr) << std::endl;
		++itr;
	}

	// input streaming
	boost::gregorian::date d3 = boost::gregorian::day_clock::local_day();
	std::stringstream ss("2004-Jan-1");
	ss >> d3;

	// date generator functions
	boost::gregorian::date d4 = boost::gregorian::day_clock::local_day();
	//boost::gregorian::date d5 = boost::gregorian::next_weekday(d4, boost::date_time::Sunday);  // calculate Sunday following d4

	// US labor day is first Monday in Sept
	boost::gregorian::nth_day_of_the_week_in_month labor_day(boost::gregorian::nth_kday_of_month::first, boost::date_time::Monday, boost::date_time::Sep);
	// calculate a specific date for 2004 from functor
	boost::gregorian::date d6 = labor_day.get_date(2004);
}

void posix_time_basic()
{
	boost::gregorian::date d(2002, boost::date_time::Feb, 1);  // an arbitrary date

#if defined(BOOST_DATE_TIME_HAS_NANOSECONDS)
	boost::posix_time::ptime t1(d, boost::posix_time::hours(5) + boost::posix_time::nanosec(100));  // date + time of day offset
#else
	boost::posix_time::ptime t1(d, boost::posix_time::hours(5) + boost::posix_time::microsec(1));  // date + time of day offset
#endif
	boost::posix_time::ptime t2 = t1 - boost::posix_time::minutes(4) + boost::posix_time::seconds(2);
	boost::posix_time::time_duration td = t2 - t1;

	std::cout << boost::posix_time::to_simple_string(t2) << " - " 
		<< boost::posix_time::to_simple_string(t1) << " = "
		<< boost::posix_time::to_simple_string(td) << std::endl;

	boost::posix_time::ptime now = boost::posix_time::second_clock::local_time();  // use the clock
	//boost::posix_time::ptime now = boost::posix_time::second_clock::universal_time();  // use the clock
	//boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();  // use the clock
	//boost::posix_time::ptime now = boost::posix_time::microsec_clock::universal_time();  // use the clock

	boost::gregorian::date today = now.date();  // Get the date part out of the time
	boost::gregorian::date tomorrow = today + boost::gregorian::date_duration(1);

	// input streaming
	std::stringstream ss("2004-Jan-1 05:21:33.20");
	ss >> t2;

	// starting at current time iterator adds by one hour
	boost::posix_time::ptime tomorrow_start(tomorrow);  // midnight

	boost::posix_time::time_iterator titr(now, boost::posix_time::hours(1));
	for (; titr < tomorrow_start; ++titr)
	{
		std::cout << (*titr) << std::endl;
	}
}
