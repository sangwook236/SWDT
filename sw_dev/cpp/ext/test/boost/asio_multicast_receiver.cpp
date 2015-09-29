#if defined(WIN32)
#define _WIN32_WINNT 0x0501
#endif

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/array.hpp>
#include <iostream>
#include <ctime>


namespace {
namespace local {

class receiver
{
public:
    receiver(boost::asio::io_service &io_service, const boost::asio::ip::address &listen_address, const boost::asio::ip::address &multicast_address, const short multicast_port)
    : socket_(io_service)
    {
        // Create the socket so that multiple may be bound to the same address.
        boost::asio::ip::udp::endpoint listen_endpoint(listen_address, multicast_port);
        socket_.open(listen_endpoint.protocol());
        socket_.set_option(boost::asio::ip::udp::socket::reuse_address(true));
        socket_.bind(listen_endpoint);

        // Join the multicast group.
        socket_.set_option(boost::asio::ip::multicast::join_group(multicast_address));

        socket_.async_receive_from(
            boost::asio::buffer(data_, max_length),
            sender_endpoint_,
            boost::bind(&receiver::handle_receive_from, this,boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred)
        );
    }

    void handle_receive_from(const boost::system::error_code &error, std::size_t bytes_recvd)
    {
        if (!error)
        {
            std::cout.write(data_, bytes_recvd);
            std::cout << std::endl;

            socket_.async_receive_from(
                boost::asio::buffer(data_, max_length),
                sender_endpoint_,
                boost::bind(&receiver::handle_receive_from, this,boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred)
            );
        }
    }

private:
    boost::asio::ip::udp::socket socket_;
    boost::asio::ip::udp::endpoint sender_endpoint_;
    enum { max_length = 1024 };
    char data_[max_length];
};

}  // namespace local
}  // unnamed namespace

// REF [file] ${BOOST_HOME}/boost_1_59_0/libs/asio/example/cpp03/multicast/receiver.cpp.
void asio_multicast_receiver()
{
    try
    {
#if 1
        // For IPv4.
        const std::string listen_address("0.0.0.0");
        const std::string multicast_address("239.255.0.1");
#else
        // For IPv6.
        const std::string listen_address("0::0");
        const std::string multicast_address("ff31::8000:1234");
#endif
        const short multicast_port = 30001;

        boost::asio::io_service io_service;
        local::receiver r(io_service, boost::asio::ip::address::from_string(listen_address), boost::asio::ip::address::from_string(multicast_address), multicast_port);

        io_service.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Boost.Asio exception: " << e.what() << "\n";
    }
}
