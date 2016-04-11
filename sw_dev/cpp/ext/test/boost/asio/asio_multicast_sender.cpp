#if defined(WIN32) || defined(_WIN32)
#define _WIN32_WINNT 0x0501
#endif

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <sstream>
#include <iostream>
#include <string>


namespace {
namespace local {

const int max_message_count = 10;

class sender
{
public:
    sender(boost::asio::io_service &io_service, const boost::asio::ip::address &multicast_address, const short multicast_port)
    : endpoint_(multicast_address, multicast_port), socket_(io_service, endpoint_.protocol()), timer_(io_service), message_count_(0)
    {
        std::ostringstream os;
        os << "Message " << message_count_++;
        message_ = os.str();

        socket_.async_send_to(
            boost::asio::buffer(message_), endpoint_,
            boost::bind(&sender::handle_send_to, this, boost::asio::placeholders::error)
        );
    }

    void handle_send_to(const boost::system::error_code &error)
    {
        if (!error && message_count_ < max_message_count)
        {
            timer_.expires_from_now(boost::posix_time::seconds(1));
            timer_.async_wait(boost::bind(&sender::handle_timeout, this, boost::asio::placeholders::error));
        }
    }

    void handle_timeout(const boost::system::error_code &error)
    {
        if (!error)
        {
            std::ostringstream os;
            os << "Message " << message_count_++;
            message_ = os.str();

            socket_.async_send_to(
                boost::asio::buffer(message_),
                endpoint_,
                boost::bind(&sender::handle_send_to, this, boost::asio::placeholders::error)
            );
        }
    }

private:
    boost::asio::ip::udp::endpoint endpoint_;
    boost::asio::ip::udp::socket socket_;
    boost::asio::deadline_timer timer_;
    int message_count_;
    std::string message_;
};

}  // namespace local
}  // unnamed namespace

// REF [file] ${BOOST_HOME}/boost_1_59_0/libs/asio/example/cpp03/multicast/sender.cpp.
void asio_multicast_sender()
{
    try
    {
#if 1
        // For IPv4.
        const std::string multicast_address("239.255.0.1");
#else
        // For IPv6.
        const std::string multicast_address("ff31::8000:1234");
#endif
        const short multicast_port = 30001;

        boost::asio::io_service io_service;
        local::sender s(io_service, boost::asio::ip::address::from_string(multicast_address), multicast_port);

        io_service.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Boost.Asio exception: " << e.what() << "\n";
    }
}
