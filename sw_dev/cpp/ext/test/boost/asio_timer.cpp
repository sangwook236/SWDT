#if defined(WIN32)
#define _WIN32_WINNT_NT4 0x0400  // Windows NT 4.0
#define _WIN32_WINNT_WIN2K 0x0500  // Windows 2000
#define _WIN32_WINNT_WINXP 0x0501  // Windows XP
#define _WIN32_WINNT_WIN7 0x0601  // Windows 7
#define _WIN32_WINNT_WIN10 0x0A00  // Windows 10
#define _WIN32_WINNT _WIN32_WINNT_WIN7
#endif

#include <boost/asio.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/bind.hpp>
#include <iostream>


namespace {
namespace local {

void print(const boost::system::error_code & /*e*/, boost::asio::deadline_timer *timer, int *count)
{
	if (*count < 5)
	{
		std::cout << *count << std::endl;
		++(*count);

        //timer->expires_from_now(boost::posix_time::seconds(1));
		timer->expires_at(timer->expires_at() + boost::posix_time::seconds(1));
		timer->async_wait(boost::bind(print, boost::asio::placeholders::error, timer, count));
	}
}

void asio_async_timer()
{
	boost::asio::io_service ioService;

	int count = 0;
	boost::asio::deadline_timer timer(ioService, boost::posix_time::seconds(1));

	timer.async_wait(boost::bind(print, boost::asio::placeholders::error, &timer, &count));
	ioService.run();

	std::cout << "final count is " << count << std::endl;
	std::cout << "io_service is terminated" << std::endl;
}

void asio_sync_timer()
{
	boost::asio::io_service ioService;

	boost::asio::deadline_timer timer(ioService, boost::posix_time::seconds(3));

	timer.wait();

	std::cout << "3 secs are elapsed !!!" << std::endl;
}

}  // namespace local
}  // unnamed namespace

void asio_timer()
{
	local::asio_async_timer();
	local::asio_sync_timer();
}
