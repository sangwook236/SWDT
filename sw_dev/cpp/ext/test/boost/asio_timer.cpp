#if defined(WIN32)
#define _WIN32_WINNT 0x0501
#endif

#include <boost/asio.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/bind.hpp>
#include <iostream>


void asio_async_timer();
void asio_sync_timer();

void asio_timer()
{
	asio_async_timer();
	asio_sync_timer();
}

namespace
{
	void print(const boost::system::error_code& /*e*/, boost::asio::deadline_timer *t, int *count)
	{
		if (*count < 5)
		{
			std::cout << *count << std::endl;
			++(*count);

			t->expires_at(t->expires_at() + boost::posix_time::seconds(1));
			t->async_wait(boost::bind(print, boost::asio::placeholders::error, t, count));
		}
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
