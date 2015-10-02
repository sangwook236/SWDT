#if defined(WIN32)
#define _WIN32_WINNT 0x0501
#endif

#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <iostream>


namespace {
namespace local {

void asio_async_udp_client()
{
	boost::asio::io_service ioService;

	// what should I do for async UDP client?
}

void asio_sync_udp_client()
{
	try
	{
		boost::asio::io_service ioService;
		boost::asio::ip::udp::resolver resolver(ioService);
		boost::asio::ip::udp::resolver::query query(boost::asio::ip::udp::v4(), "localhost", "daytime");  // use a service name
		//boost::asio::ip::udp::resolver::query query(boost::asio::ip::udp::v4(), "localhost", "30001");  // use a port number
		boost::asio::ip::udp::endpoint receiver_endpoint = *resolver.resolve(query);
		boost::asio::ip::udp::socket socket(ioService);
		socket.open(boost::asio::ip::udp::v4());

		boost::array<char, 1> send_buf  = { 0 };
		socket.send_to(boost::asio::buffer(send_buf), receiver_endpoint);

		boost::array<char, 128> recv_buf;
		boost::asio::ip::udp::endpoint sender_endpoint;
		const size_t len = socket.receive_from(boost::asio::buffer(recv_buf), sender_endpoint);

		std::cout.write(recv_buf.data(), len);
	}
	catch (const std::exception &e)
	{
		std::cerr << "Boost.Asio exception: " << e.what() << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

void asio_udp_client()
{
	//local::asio_async_udp_client();
	local::asio_sync_udp_client();
}
