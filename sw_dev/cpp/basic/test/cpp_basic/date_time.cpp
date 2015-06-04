#include <sstream>
#include <locale>
#include <iostream>
#include <iomanip>
#include <ctime>


/*
	The C++ standard library does not provide a proper date type.
	C++ inherits the structs and functions for date and time manipulation from C.
	To access date and time related functions and structures, you would need to include <ctime> header file in your C++ program.

	There are four time-related types: clock_t, time_t, size_t, and tm.
	The types clock_t, size_t and time_t are capable of representing the system time and date as some sort of integer.
	The structure type tm holds the date and time in the form of a C structure having the following elements:
		struct tm {
			int tm_sec;   // seconds of minutes from 0 to 61.
			int tm_min;   // minutes of hour from 0 to 59.
			int tm_hour;  // hours of day from 0 to 24.
			int tm_mday;  // day of month from 1 to 31.
			int tm_mon;   // month of year from 0 to 11.
			int tm_year;  // year since 1900.
			int tm_wday;  // days since sunday.
			int tm_yday;  // days since January 1st.
			int tm_isdst; // hours of daylight savings time.
		};

	Following are the important functions, which we use while working with date and time in C or C++.
	All these functions are part of standard C and C++ library and you can check their detail using reference to C++ standard library given below.
		1. time_t time(time_t *time);
			This returns the current calendar time of the system in number of seconds elapsed since January 1, 1970.
			If the system has no time, .1 is returned.
		2. char * ctime(const time_t *time);
			This returns a pointer to a string of the form : day month year hours : minutes : seconds year\n\0.
		3. struct tm * localtime(const time_t *time);
			This returns a pointer to the tm structure representing local time.
			A value of .1 is returned if the time is not available.
		4. struct tm * gmtime(const time_t *time);
			This returns a pointer to the time in the form of a tm structure.
			The time is represented in Coordinated Universal Time(UTC), which is essentially Greenwich Mean Time(GMT).
		5. time_t mktime(struct tm *time);
			This returns the calendar - time equivalent of the time found in the structure pointed to by time.
			mktime() can compute the values of tm_mday and tm_yday from other members; it isn't designed to compute the values of other members from those fields.
			This function performs the reverse translation that localtime does.
		6. char * asctime(const struct tm *time);
			This returns a pointer to a string that contains the information stored in the structure pointed to by time converted into the form : day month date hours : minutes : seconds year\n\0.
		7. clock_t clock();
			This returns a value that approximates the amount of time the calling program has been running.
		8. double difftime(time_t time2, time_t time1);
			This function calculates the difference in seconds between time1 and time2.
		9. size_t strftime();
			This function can be used to format date and time a specific format.
*/

namespace {
namespace local {

void wait(const int seconds)
{
	const clock_t endwait = std::clock() + seconds * CLOCKS_PER_SEC;
	while (std::clock() < endwait) ;
}

void clock_example()
{
	std::cout << "starting count down..." << std::endl;;
	for (int n = 5; n >= 0; --n)
	{
		std::cout << '\t' << n << std::endl;;
		wait(1);
	}
}

void UTC_time_example()
{
	const int MST = -7;
	const int UTC = 0;
	const int CCT = +8;

#if 0
	time_t rawtime;
	std::time(&rawtime);  // get the current time, [sec].
#else
	const time_t rawtime = std::time(NULL);  // get the current time, [sec].
	if ((time_t)-1 == rawtime)
	{
		std::cerr << "could not retrieve the calendar time." << std::endl;
		return;
	}
#endif

	// convert time_t to tm as UTC time.
	const struct tm *timeinfo = std::gmtime(&rawtime);

	std::cout << "current time around the world:" << std::endl;
	std::cout << "\tPhoenix, AZ (U.S.):  " << std::setw(2) << (timeinfo->tm_hour + MST) % 24 << ":" << std::setfill('0') << std::setw(2) << timeinfo->tm_min << std::endl;
	std::cout << "\tReykjavik (Iceland): " << std::setw(2) << (timeinfo->tm_hour + UTC) % 24 << ":" << std::setfill('0') << std::setw(2) << timeinfo->tm_min << std::endl;
	std::cout << "\tBeijing (China):     " << std::setw(2) << (timeinfo->tm_hour + CCT) % 24 << ":" << std::setfill('0') << std::setw(2) << timeinfo->tm_min << std::endl;
}

void local_time_example()
{
#if 0
	time_t rawtime;
	std::time(&rawtime);  // get the current time, [sec].
#else
	const time_t rawtime = std::time(NULL);  // get the current time, [sec].
	if ((time_t)-1 == rawtime)
	{
		std::cerr << "could not retrieve the calendar time." << std::endl;
		return;
	}
#endif

	// convert time_t to tm as local time.
	const struct tm *timeinfo = std::localtime(&rawtime);

	std::cout << "current local time and date:" << std::endl;
	// display date time using time_t.
	std::cout << '\t' << std::ctime(&rawtime);
	// display date time using struct tm.
	std::cout << '\t' << std::asctime(timeinfo);

	// format time to string.
	char buffer[80];
	const std::size_t strLen = std::strftime(buffer, 80, "%I:%M%p.", timeinfo);
	std::cout << '\t' << buffer << std::endl;

	//
	const int year = 2000;
	const int month = 5;
	const int day = 20;

	struct tm *timeinfo2 = std::localtime(&rawtime);
	timeinfo2->tm_year = year - 1900;
	timeinfo2->tm_mon = month - 1;
	timeinfo2->tm_mday = day;

	// convert struct tm to time_t.
	const time_t rawtime2 = std::mktime(timeinfo2);

	const char *weekday[] = { "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
	std::cout << "that day was " << weekday[timeinfo2->tm_wday] << std::endl;
}

void mktime_example()
{
#if 0
	time_t rawtime;
	std::time(&rawtime);  // get the current time, [sec].
#else
	const time_t rawtime = std::time(NULL);  // get the current time, [sec].
	if ((time_t)-1 == rawtime)
	{
		std::cerr << "could not retrieve the calendar time." << std::endl;
		return;
	}
#endif

	// convert time_t to tm as local time.
	const struct tm *timeinfo = std::localtime(&rawtime);

	{
		struct tm when = { 0, };
		when.tm_sec = sec;  // seconds of minutes from 0 to 61.
		when.tm_min = minute;  // minutes of hour from 0 to 59.
		when.tm_hour = hour;  // hours of day from 0 to 24.
		when.tm_mday = day;  // day of month from 1 to 31.
		when.tm_mon = mon;  // month of year from 0 to 11.
		when.tm_year = year;  // year since 1900.
		//when.tm_wday = ;  // days since sunday.
		//when.tm_yday = ;  // days since January 1st.
		//when.tm_isdst = timeinfo->tm_isdst;  // hours of daylight savings time.
		when.tm_isdst = -1;  // hours of daylight savings time.

		const time_t seconds = std::mktime(&when);  // in the local time.

		// automatically compute the below values.
		std::cout << "when.tm_wday  = " << when.tm_wday << std::endl;
		std::cout << "when.tm_yday  = " << when.tm_yday << std::endl;
		std::cout << "when.tm_isdst = " << when.tm_isdst << std::endl;
	}

	{
		// [ref] http://pubs.opengroup.org/onlinepubs/009695399/basedefs/xbd_chap04.html#tag_04_14
		//	If the year is <1970 or the value is negative, the relationship is undefined.
		//	If the year is >=1970 and the value is non-negative, the value is related to a Coordinated Universal Time name according to the C-language expression, where tm_sec, tm_min, tm_hour, tm_yday, and tm_year are all integer types:
		//		tm_sec + tm_min*60 + tm_hour*3600 + tm_yday*86400 + (tm_year-70)*31536000 + ((tm_year-69)/4)*86400 - ((tm_year-1)/100)*86400 + ((tm_year+299)/400)*86400

        const int year = 115 /*since 1900*/, mon = 5 /*June*/, day = 1;
        const int yday = 31 + 28 + 31 + 30 + 31 + day - 1;  // 151.
        const int hour = 14, minute = 20, sec = 15;

		struct tm when = { 0, };
		when.tm_sec = sec;  // seconds of minutes from 0 to 61.
		when.tm_min = minute;  // minutes of hour from 0 to 59.
		when.tm_hour = hour;  // hours of day from 0 to 24.
		when.tm_mday = day;  // day of month from 1 to 31.
		when.tm_mon = mon;  // month of year from 0 to 11.
		when.tm_year = year;  // year since 1900.
		//when.tm_wday = ;  // days since sunday.
		//when.tm_yday = ;  // days since January 1st.
		//when.tm_isdst = timeinfo->tm_isdst;  // hours of daylight savings time.
		when.tm_isdst = -1;  // hours of daylight savings time.

		const time_t seconds = std::mktime(&when);  // in the local time.
		std::cout << "seconds obtained by mktime() = " << seconds << std::endl;

		{
		    const struct tm *lt = std::localtime(&seconds);
		    std::cout << "\tlocaltime->tm_sec   = " << lt->tm_sec << std::endl;
		    std::cout << "\tlocaltime->tm_min   = " << lt->tm_min << std::endl;
		    std::cout << "\tlocaltime->tm_hour  = " << lt->tm_hour << std::endl;  // 14. (O)
		    std::cout << "\tlocaltime->tm_mday  = " << lt->tm_mday << std::endl;
		    std::cout << "\tlocaltime->tm_mon   = " << lt->tm_mon << std::endl;
		    std::cout << "\tlocaltime->tm_year  = " << lt->tm_year << std::endl;
		    std::cout << "\tlocaltime->tm_wday  = " << lt->tm_wday << std::endl;
		    std::cout << "\tlocaltime->tm_yday  = " << lt->tm_yday << std::endl;
		    std::cout << "\tlocaltime->tm_isdst = " << lt->tm_isdst << std::endl;  // 1. (?)

		    const struct tm *utc = std::gmtime(&seconds);
		    std::cout << "\tutc->tm_sec   = " << utc->tm_sec << std::endl;
		    std::cout << "\tutc->tm_min   = " << utc->tm_min << std::endl;
		    std::cout << "\tutc->tm_hour  = " << utc->tm_hour << std::endl;  // 21. (X)
		    std::cout << "\tutc->tm_mday  = " << utc->tm_mday << std::endl;
		    std::cout << "\tutc->tm_mon   = " << utc->tm_mon << std::endl;
		    std::cout << "\tutc->tm_year  = " << utc->tm_year << std::endl;
		    std::cout << "\tutc->tm_wday  = " << utc->tm_wday << std::endl;
		    std::cout << "\tutc->tm_yday  = " << utc->tm_yday << std::endl;
		    std::cout << "\tutc->tm_isdst = " << utc->tm_isdst << std::endl;  // 0.
		}

        const time_t seconds_gmt_calculated = (time_t)(sec + minute*60 + hour*3600 + yday*86400 + (year-70)*31536000 + ((year-69)/4)*86400 - ((year-1)/100)*86400 + ((year+299)/400)*86400);  // in UTC.
		std::cout << "seconds in UTC calculated by a equation = " << seconds_calculated << std::endl;

        struct tm *seconds_ti_calculated = std::gmtime(&seconds_gmt_calculated);
        seconds_ti_calculated->tm_isdst = -1;
        const time_t seconds_localtime_calculated = std::mktime(seconds_ti_calculated);
		std::cout << "seconds in UTC calculated by a equation = " << seconds_localtime_calculated << std::endl;

		{
            std::cout << "seconds_ti_calculated->tm_sec   = " << seconds_ti_calculated->tm_sec << std::endl;
            std::cout << "seconds_ti_calculated->tm_min   = " << seconds_ti_calculated->tm_min << std::endl;
            std::cout << "seconds_ti_calculated->tm_hour  = " << seconds_ti_calculated->tm_hour << std::endl;  // 14. (O)
            std::cout << "seconds_ti_calculated->tm_mday  = " << seconds_ti_calculated->tm_mday << std::endl;
            std::cout << "seconds_ti_calculated->tm_mon   = " << seconds_ti_calculated->tm_mon << std::endl;
            std::cout << "seconds_ti_calculated->tm_year  = " << seconds_ti_calculated->tm_year << std::endl;
            std::cout << "seconds_ti_calculated->tm_wday  = " << seconds_ti_calculated->tm_wday << std::endl;
            std::cout << "seconds_ti_calculated->tm_yday  = " << seconds_ti_calculated->tm_yday << std::endl;
            std::cout << "seconds_ti_calculated->tm_isdst = " << seconds_ti_calculated->tm_isdst << std::endl;  // 1. (?)

		}

		{
		    const struct tm *lt = std::localtime(&seconds_calculated);
		    std::cout << "\tlocaltime->tm_sec   = " << lt->tm_sec << std::endl;
		    std::cout << "\tlocaltime->tm_min   = " << lt->tm_min << std::endl;
		    std::cout << "\tlocaltime->tm_hour  = " << lt->tm_hour << std::endl;  // 7. (X)
		    std::cout << "\tlocaltime->tm_mday  = " << lt->tm_mday << std::endl;
		    std::cout << "\tlocaltime->tm_mon   = " << lt->tm_mon << std::endl;
		    std::cout << "\tlocaltime->tm_year  = " << lt->tm_year << std::endl;
		    std::cout << "\tlocaltime->tm_wday  = " << lt->tm_wday << std::endl;
		    std::cout << "\tlocaltime->tm_yday  = " << lt->tm_yday << std::endl;
		    std::cout << "\tlocaltime->tm_isdst = " << lt->tm_isdst << std::endl;  // 1. (?)

		    const struct tm *utc = std::gmtime(&seconds_calculated);
		    std::cout << "\tutc->tm_sec   = " << utc->tm_sec << std::endl;
		    std::cout << "\tutc->tm_min   = " << utc->tm_min << std::endl;
		    std::cout << "\tutc->tm_hour  = " << utc->tm_hour << std::endl;  // 14. (O)
		    std::cout << "\tutc->tm_mday  = " << utc->tm_mday << std::endl;
		    std::cout << "\tutc->tm_mon   = " << utc->tm_mon << std::endl;
		    std::cout << "\tutc->tm_year  = " << utc->tm_year << std::endl;
		    std::cout << "\tutc->tm_wday  = " << utc->tm_wday << std::endl;
		    std::cout << "\tutc->tm_yday  = " << utc->tm_yday << std::endl;
		    std::cout << "\tutc->tm_isdst = " << utc->tm_isdst << std::endl;  // 0.
		}
	}
}

void time_diff_example()
{
	time_t start, finish;

	std::time(&start);
	//const time_t start = std::time(NULL);
	{
		double result;
		// multiplying 2 floating point numbers 500 million times.
		for (long loop = 0; loop < 500000000; ++loop)
			result = 3.63 * 5.27;
	}
	std::time(&finish);
	//const time_t finish = std::time(NULL);

	const double elapsed_time = std::difftime(finish, start);
	std::cout << "it takes " << elapsed_time << " seconds" << std::endl;
}

void date_order_example()
{
	std::locale loc;
	const std::time_get<char>::dateorder order = std::use_facet<std::time_get<char> >(loc).date_order();

	switch (order)
	{
	case std::time_get<char>::no_order:
		std::cout << "no_order" << std::endl;
		break;
	case std::time_get<char>::dmy:
		std::cout << "dmy" << std::endl;
		break;
	case std::time_get<char>::mdy:
		std::cout << "mdy" << std::endl;
		break;
	case std::time_get<char>::ymd:
		std::cout << "ymd" << std::endl;
		break;
	case std::time_get<char>::ydm:
		std::cout << "ydm" << std::endl;
		break;
	}
}

void time_example()
{
	std::locale loc;  // "C" locale.

	// get time_get facet.
	const std::time_get<char> &tmget = std::use_facet<std::time_get<char> >(loc);

	std::ios::iostate state;
	std::istringstream iss("07:30:00");
	std::istreambuf_iterator<char> itbegin(iss);  // beginning of iss.
	std::istreambuf_iterator<char> itend;  // end-of-stream.
	struct tm when;

	tmget.get_time(itbegin, itend, iss, state, &when);

	std::cout << "hour: " << when.tm_hour << std::endl;
	std::cout << "min: " << when.tm_min << std::endl;
	std::cout << "sec: " << when.tm_sec << std::endl;
}

void date_example()
{
	std::locale loc;  // "C" locale.

	// get time_get facet.
	const std::time_get<char> &tmget = std::use_facet<std::time_get<char> >(loc);

	std::ios::iostate state;
	std::istringstream iss("01/02/03");
	std::istreambuf_iterator<char> itbegin(iss);  // beginning of iss.
	std::istreambuf_iterator<char> itend;  // end-of-stream.
	struct tm when;

	tmget.get_date(itbegin, itend, iss, state, &when);

	std::cout << "year: " << when.tm_year << std::endl;
	std::cout << "month: " << when.tm_mon << std::endl;
	std::cout << "day: " << when.tm_mday << std::endl;
}

void weekday_example()
{
	std::locale loc;  // "C" locale.

	// get time_get facet.
	const std::time_get<char> &tmget = std::use_facet<std::time_get<char> >(loc);

	std::ios::iostate state;
	std::istringstream iss("Friday");
	std::istreambuf_iterator<char> itbegin(iss);  // beginning of iss.
	std::istreambuf_iterator<char> itend;  // end-of-stream.
	struct tm when;

	tmget.get_weekday(itbegin, itend, iss, state, &when);

	std::cout << "weekday: " << when.tm_wday << std::endl;
}

void month_example()
{
	std::locale loc;  // "C" locale.

	// get time_get facet.
	const std::time_get<char> &tmget = std::use_facet<std::time_get<char> >(loc);

	std::ios::iostate state;
	std::istringstream iss("August");
	std::istreambuf_iterator<char> itbegin(iss);  // beginning of iss.
	std::istreambuf_iterator<char> itend;  // end-of-stream.
	struct tm when;

	tmget.get_monthname(itbegin, itend, iss, state, &when);

	std::cout << "month: " << (when.tm_mon + 1) << std::endl;
}

void year_example()
{
	std::locale loc;  // "C" locale.

	// get time_get facet.
	const std::time_get<char> &tmget = std::use_facet<std::time_get<char> >(loc);

	std::ios::iostate state;
	std::istringstream iss("2009");
	std::istreambuf_iterator<char> itbegin(iss);  // beginning of iss.
	std::istreambuf_iterator<char> itend;  // end-of-stream.
	struct tm when;

	tmget.get_year(itbegin, itend, iss, state, &when);

	std::cout << "year: " << (when.tm_year + 1900) << std::endl;
}

/**
 * @brief Convert seconds into date and time in the local time.
 *
 * @param [in] seconds  the number of seconds elapsed since January 1, 1970.
 * @param [out] year  year. >= 1900.
 * @param [out] mon  month of year from 1 to 12.
 * @param [out] day  day of month from 1 to 31.
 * @param [out] hour  hours of day from 0 to 24.
 * @param [out] minute  minutes of hour from 0 to 59.
 * @param [out] sec  seconds of minutes from 0 to 61.
 */
void seconds2datetime(const time_t seconds, int &year, int &mon, int &day, int &hour, int &minute, int &sec)
{
	struct tm *when = std::localtime(&seconds);
	sec = when->tm_sec;  // seconds of minutes from 0 to 61.
	minute = when->tm_min;  // minutes of hour from 0 to 59.
	hour = when->tm_hour;  // hours of day from 0 to 24.
	day = when->tm_mday;  // day of month from 1 to 31.
	mon = when->tm_mon + 1;  // month of year from 0 to 11.
	year = when->tm_year + 1900;  // year since 1900.
}

/**
 * @brief Convert date and time into seconds in the local time.
 *
 * @param [in] year  year. >= 1900.
 * @param [in] mon  month of year from 1 to 12.
 * @param [in] day  day of month from 1 to 31.
 * @param [in] hour  hours of day from 0 to 24.
 * @param [in] minute  minutes of hour from 0 to 59.
 * @param [in] sec  seconds of minutes from 0 to 61.
 * @param [out] seconds  the number of seconds elapsed since January 1, 1970.
 */
void datetime2seconds(const int year, const int mon, const int day, const int hour, const int minute, const int sec, time_t &seconds)
{
    // [ref] http://pubs.opengroup.org/onlinepubs/009695399/basedefs/xbd_chap04.html#tag_04_14

    //const time_t now = std::time(NULL);  // get the current time, [sec].
    //struct tm *ti = std::localtime(&now);

	struct tm when = { 0, };
	when.tm_sec = sec;  // seconds of minutes from 0 to 61.
	when.tm_min = minute;  // minutes of hour from 0 to 59.
	when.tm_hour = hour;  // hours of day from 0 to 24.
	when.tm_mday = day;  // day of month from 1 to 31.
	when.tm_mon = mon - 1;  // month of year from 0 to 11.
	when.tm_year = year - 1900;  // year since 1900.
	//when.tm_wday = ;  // days since sunday.
	//when.tm_yday = ;  // days since January 1st.
	//when.tm_isdst = ti->tm_isdst;  // hours of daylight savings time.
	when.tm_isdst = -1;  // hours of daylight savings time.

	seconds = std::mktime(&when);  // in the local time.
}
	
}  // namespace local
}  // unnamed namespace

void date_time()
{
	local::clock_example();
	local::UTC_time_example();
	local::local_time_example();
	local::mktime_example();

	local::time_diff_example();

	//
	local::date_order_example();

	local::time_example();
	local::date_example();
	local::weekday_example();
	local::month_example();
	local::year_example();

	//
	local::seconds2datetime_test();
	local::datetime2seconds_test();
}
