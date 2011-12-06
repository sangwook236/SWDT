#include "stdafx.h"

#define _WIN32_WINNT 0x0501

#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <iostream>


void asio_async_udp_client();
void asio_sync_udp_client();

void asio_udp_client()
{
	//asio_async_udp_client();
	asio_sync_udp_client();
}

void asio_async_udp_client()
{
	boost::asio::io_service ioService;
}

void asio_sync_udp_client()
{
	try
	{
		boost::asio::io_service ioService;
		boost::asio::ip::udp::resolver resolver(ioService);
		boost::asio::ip::udp::resolver::query query(boost::asio::ip::udp::v4(), "localhost", "daytime");  // use a service name
		//boost::asio::ip::udp::resolver::query query(boost::asio::ip::udp::v4(), "localhost", "13");  // use a port number
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
		std::cerr << e.what() << std::endl;
	}
}
