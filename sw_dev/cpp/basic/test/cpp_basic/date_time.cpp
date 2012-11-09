#include <sstream>
#include <locale>
#include <iostream>
#include <iomanip>


namespace {
namespace local {

void wait(const int seconds)
{
	const clock_t endwait = std::clock() + seconds * CLOCKS_PER_SEC;
	while (std::clock() < endwait) ;
}

void date_time__clock()
{
	std::cout << "starting count down..." << std::endl;;
	for (int n = 5; n >= 0; --n)
	{
		std::cout << '\t' << n << std::endl;;
		wait(1);
	}
}

void date_time__UTC_time()
{
	const int MST = -7;
	const int UTC = 0;
	const int CCT = +8;

	time_t rawtime;
	// get current time
	std::time(&rawtime);
	// convert time_t to tm as UTC time
	const struct tm *timeinfo = std::gmtime(&rawtime);

	std::cout << "current time around the world:" << std::endl;
	std::cout << "\tPhoenix, AZ (U.S.):  " << std::setw(2) << (timeinfo->tm_hour+MST) % 24 << ":" << std::setfill('0') << std::setw(2) << timeinfo->tm_min << std::endl;
	std::cout << "\tReykjavik (Iceland): " << std::setw(2) << (timeinfo->tm_hour+UTC) % 24 << ":" << std::setfill('0') << std::setw(2) << timeinfo->tm_min << std::endl;
	std::cout << "\tBeijing (China):     " << std::setw(2) << (timeinfo->tm_hour+CCT) % 24 << ":" << std::setfill('0') << std::setw(2) << timeinfo->tm_min << std::endl;
}

void date_time__local_time()
{
	time_t rawtime;
	// get current time
	std::time(&rawtime);
	// convert time_t to tm as local time
	const struct tm *timeinfo = std::localtime(&rawtime);
	std::cout << "current local time and date:" << std::endl;

	// convert time_t value to string
	std::cout << '\t' << std::ctime(&rawtime);

	// convert tm structure to string
	std::cout << '\t' << std::asctime(timeinfo);

	// format time to string
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

	// convert tm structure to time_t
	std::mktime(timeinfo2);

	const char * weekday[] = { "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
	std::cout << "that day is a " << weekday[timeinfo2->tm_wday] << std::endl;
}

void date_time__time_diff()
{
	time_t start, finish;

	time(&start);
	double result;
	// multiplying 2 floating point numbers 500 million times
	for (long loop = 0; loop < 500000000; ++loop)
		result = 3.63 * 5.27; 
	time(&finish);

	const double elapsed_time = difftime(finish, start);
	std::cout << "it takes %6.0f seconds " << elapsed_time << std::endl;
}

void date_time__date_order()
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

void date_time__time()
{
	std::locale loc;  // "C" locale

	// get time_get facet:
	const std::time_get<char>& tmget = std::use_facet<std::time_get<char> >(loc);

	std::ios::iostate state;
	std::istringstream iss("07:30:00");
	std::istreambuf_iterator<char> itbegin(iss);  // beginning of iss
	std::istreambuf_iterator<char> itend;         // end-of-stream
	tm when;

	tmget.get_time(itbegin, itend, iss, state, &when);

	std::cout << "hour: " << when.tm_hour << std::endl;
	std::cout << "min: " << when.tm_min << std::endl;
	std::cout << "sec: " << when.tm_sec << std::endl;
}

void date_time__date()
{
	std::locale loc;  // "C" locale

	// get time_get facet:
	const std::time_get<char>& tmget = std::use_facet<std::time_get<char> >(loc);

	std::ios::iostate state;
	std::istringstream iss("01/02/03");
	std::istreambuf_iterator<char> itbegin(iss);  // beginning of iss
	std::istreambuf_iterator<char> itend;         // end-of-stream
	tm when;

	tmget.get_date(itbegin, itend, iss, state, &when);

	std::cout << "year: " << when.tm_year << std::endl;
	std::cout << "month: " << when.tm_mon << std::endl;
	std::cout << "day: " << when.tm_mday << std::endl;
}

void date_time__weekday()
{
	std::locale loc;  // "C" locale

	// get time_get facet:
	const std::time_get<char>& tmget = std::use_facet<std::time_get<char> >(loc);

	std::ios::iostate state;
	std::istringstream iss("Friday");
	std::istreambuf_iterator<char> itbegin(iss);  // beginning of iss
	std::istreambuf_iterator<char> itend;         // end-of-stream
	tm when;

	tmget.get_weekday(itbegin, itend, iss, state, &when);

	std::cout << "weekday: " << when.tm_wday << std::endl;
}

void date_time__month()
{
	std::locale loc;  // "C" locale

	// get time_get facet:
	const std::time_get<char>& tmget = std::use_facet<std::time_get<char> >(loc);

	std::ios::iostate state;
	std::istringstream iss("August");
	std::istreambuf_iterator<char> itbegin(iss);  // beginning of iss
	std::istreambuf_iterator<char> itend;         // end-of-stream
	tm when;

	tmget.get_monthname(itbegin, itend, iss, state, &when);

	std::cout << "month: " << (when.tm_mon + 1) << std::endl;
}

void date_time__year()
{
	std::locale loc;  // "C" locale

	// get time_get facet:
	const std::time_get<char>& tmget = std::use_facet<std::time_get<char> >(loc);

	std::ios::iostate state;
	std::istringstream iss("2009");
	std::istreambuf_iterator<char> itbegin(iss);  // beginning of iss
	std::istreambuf_iterator<char> itend;         // end-of-stream
	tm when;

	tmget.get_year(itbegin, itend, iss, state, &when);

	std::cout << "tm_year: " << when.tm_year << std::endl;
}
	
}  // namespace local
}  // unnamed namespace

void test_date_time()
{
	local::date_time__clock();
	local::date_time__UTC_time();
	local::date_time__local_time();
	local::date_time__time_diff();

	//
	local::date_time__date_order();
	local::date_time__time();
	local::date_time__date();
	local::date_time__weekday();
	local::date_time__month();
	local::date_time__year();
}
