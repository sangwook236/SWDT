#if defined(WIN32)
#define _WIN32_WINNT 0x0501
#endif

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <iostream>


namespace {
namespace local {

class sender
{
public:
    sender(boost::asio::io_service &io_service, const short broadcast_port, const std::string &message = std::string())
    : endpoint_(boost::asio::ip::address_v4::broadcast(), broadcast_port), socket_(io_service), timer_(io_service)
    {
        boost::system::error_code error;
        socket_.open(boost::asio::ip::udp::v4(), error);
        //socket_.open(endpoint_.protocol(), error);
        if (!error)
        {
            socket_.set_option(boost::asio::ip::udp::socket::reuse_address(true));
            socket_.set_option(boost::asio::socket_base::broadcast(true));

            if (!message.empty())
                send_to(message);
        }
        else
            std::cerr << "cannot open socket" << std::endl;
    }

public:
    void send_to(const std::string &message)
    {
        // TODO [delete] >> for display.
        std::cout << "start sending message ..." << std::endl;

#if 0
        // Sync.
        socket_.send_to(message, endpoint_);

        // TODO [delete] >> for display.
        std::cout << "end sending message ..." << std::endl;
#else
        // Async.
        socket_.async_send_to(
            boost::asio::buffer(message), endpoint_,
            boost::bind(&sender::handle_send_to, this, boost::asio::placeholders::error)
        );

        // TODO [delete] >> for display.
        std::cout << "keep async-sending message ..." << std::endl;
#endif
    }

private:
    void handle_send_to(const boost::system::error_code &error)
    {
        if (!error)
        {
            timer_.expires_from_now(boost::posix_time::seconds(1));
            timer_.async_wait(boost::bind(&sender::handle_timeout, this, boost::asio::placeholders::error));
        }
    }

    void handle_timeout(const boost::system::error_code &error)
    {
        if (!error)
        {
/*
            socket_.async_send_to(
                boost::asio::buffer(message),
                endpoint_,
                boost::bind(&sender::handle_send_to, this, boost::asio::placeholders::error)
            );
*/

            // TODO [delete] >> for display.
            std::cout << "end sending message ..." << std::endl;
        }
        else
        {
            std::cerr << "error value = " << error.value() << ", error message = " << error.message() << std::endl;
        }
    }

private:
    boost::asio::ip::udp::endpoint endpoint_;
    boost::asio::ip::udp::socket socket_;
    boost::asio::deadline_timer timer_;
};

}  // namespace local
}  // unnamed namespace

// REF [site] >> http://stackoverflow.com/questions/9310231/boostasio-udp-broadcasting.
void asio_udp_broadcast()
{
    // NOTE [caution] >> Use asio_udp_server as a server.

	try
	{
		const short broadcast_port = 30001;
		const std::string message("!@#$% a UDP broadcast test message ^&*()");

		boost::asio::io_service ioService;
        local::sender s(ioService, broadcast_port, message);
        //s.send_to(message);

        ioService.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << "Boost.Asio exception: " << e.what () << std::endl;
	}
}
