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

	time_t rawtime;
	// get current time.
	std::time(&rawtime);

	// convert time_t to tm as UTC time.
	const struct tm *timeinfo = std::gmtime(&rawtime);

	std::cout << "current time around the world:" << std::endl;
	std::cout << "\tPhoenix, AZ (U.S.):  " << std::setw(2) << (timeinfo->tm_hour + MST) % 24 << ":" << std::setfill('0') << std::setw(2) << timeinfo->tm_min << std::endl;
	std::cout << "\tReykjavik (Iceland): " << std::setw(2) << (timeinfo->tm_hour + UTC) % 24 << ":" << std::setfill('0') << std::setw(2) << timeinfo->tm_min << std::endl;
	std::cout << "\tBeijing (China):     " << std::setw(2) << (timeinfo->tm_hour + CCT) % 24 << ":" << std::setfill('0') << std::setw(2) << timeinfo->tm_min << std::endl;
}

void local_time_example()
{
	time_t rawtime;
	// get current time.
	std::time(&rawtime);

	// convert time_t to tm as local time.
	const struct tm *timeinfo = std::localtime(&rawtime);

	std::cout << "current local time and date:" << std::endl;
	// display date time using time_t.
	std::cout << '\t' << std::ctime(&rawtime);
	// display date time using struct tm.
	std::cout << '\t' << std::asctime(timeinfo);

	// format time to string.
	char buffer[80];
	std::strftime(buffer, 80, "%I:%M%p.", timeinfo);
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
	std::cout << "that day is a " << weekday[timeinfo2->tm_wday] << std::endl;
}

void time_diff_example()
{
	time_t start, finish;

	std::time(&start);
	{
		double result;
		// multiplying 2 floating point numbers 500 million times.
		for (long loop = 0; loop < 500000000; ++loop)
			result = 3.63 * 5.27;
	}
	std::time(&finish);

	const double elapsed_time = std::difftime(finish, start);
	std::cout << "it takes " << elapsed_time << " seconds " << std::endl;
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
	const std::time_get<char>& tmget = std::use_facet<std::time_get<char> >(loc);

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
	const std::time_get<char>& tmget = std::use_facet<std::time_get<char> >(loc);

	std::ios::iostate state;
	std::istringstream iss("01/02/03");
	std::istreambuf_iterator<char> itbegin(iss);  // beginning of iss.
	std::istreambuf_iterator<char> itend; // end-of-stream.
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
	const std::time_get<char>& tmget = std::use_facet<std::time_get<char> >(loc);

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
	const std::time_get<char>& tmget = std::use_facet<std::time_get<char> >(loc);

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
	const std::time_get<char>& tmget = std::use_facet<std::time_get<char> >(loc);

	std::ios::iostate state;
	std::istringstream iss("2009");
	std::istreambuf_iterator<char> itbegin(iss);  // beginning of iss.
	std::istreambuf_iterator<char> itend;  // end-of-stream.
	struct tm when;

	tmget.get_year(itbegin, itend, iss, state, &when);

	std::cout << "year: " << when.tm_year << std::endl;
}
	
}  // namespace local
}  // unnamed namespace

void date_time()
{
	local::clock_example();
	local::UTC_time_example();
	local::local_time_example();

	local::time_diff_example();

	//
	local::date_order_example();

	local::time_example();
	local::date_example();
	local::weekday_example();
	local::month_example();
	local::year_example();
}
