#include <boost/asio.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <iostream>


namespace {
namespace local {

class printer
{
public:
	printer(boost::asio::io_service& io)
	: strand_(io),
		timer1_(io, boost::posix_time::seconds(1)),
		timer2_(io, boost::posix_time::seconds(1)),
		count_(0)
	{
		timer1_.async_wait(strand_.wrap(boost::bind(&printer::print1, this)));
		timer2_.async_wait(strand_.wrap(boost::bind(&printer::print2, this)));
	}

	~printer()
	{
		std::cout << "final count is " << count_ << std::endl;
	}

	void print1()
	{
		if (count_ < 10)
		{
			std::cout << "Timer 1: " << count_ << std::endl;
			++count_;

			timer1_.expires_at(timer1_.expires_at() + boost::posix_time::seconds(1));
			timer1_.async_wait(strand_.wrap(boost::bind(&printer::print1, this)));
		}
	}

	void print2()
	{
		if (count_ < 10)
		{
			std::cout << "Timer 2: " << count_ << std::endl;
			++count_;

			timer2_.expires_at(timer2_.expires_at() + boost::posix_time::seconds(1));
			timer2_.async_wait(strand_.wrap(boost::bind(&printer::print2, this)));
		}
	}

private:
	boost::asio::io_context::strand strand_;
	boost::asio::deadline_timer timer1_;
	boost::asio::deadline_timer timer2_;
	int count_;
};

}  // namespace local
}  // unnamed namespace

void asio_synchronizing_handler()
{
	boost::asio::io_service ioService;
	local::printer p(ioService);

	boost::thread t(boost::bind(&boost::asio::io_service::run, &ioService));
	ioService.run();

	t.join();
}
