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

std::string make_daytime_string()
{
	std::time_t now = std::time(0);
	return std::ctime(&now);
}

class udp_server
{
public:
	enum { port_num = 13 };

public:
	udp_server(boost::asio::io_service& io_service)
	: socket_(io_service, boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), port_num))
	{
		start_receive();
	}

private:
	void start_receive()
	{
		socket_.async_receive_from(
			boost::asio::buffer(recv_buffer_),
			remote_endpoint_,
			boost::bind(&udp_server::handle_receive, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred)
		);
	}
	void handle_receive(const boost::system::error_code& error,	std::size_t /*bytes_transferred*/)
	{
		if (!error || error == boost::asio::error::message_size)
		{
			boost::shared_ptr<std::string> message(new std::string(make_daytime_string()));
			socket_.async_send_to(
				boost::asio::buffer(*message),
				remote_endpoint_,
				boost::bind(&udp_server::handle_send, this, message, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred)
			);

			start_receive();
		}
	}
	void handle_send(boost::shared_ptr<std::string> /*message*/, const boost::system::error_code& /*error*/, std::size_t /*bytes_transferred*/)
	{
	}

private:
	boost::asio::ip::udp::socket socket_;
	boost::asio::ip::udp::endpoint remote_endpoint_;
	boost::array<char, 1> recv_buffer_;
};

void asio_async_udp_server()
{
	try
	{
		boost::asio::io_service ioService;
		udp_server server(ioService);

		ioService.run();
	}
	catch (std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}

void asio_sync_udp_server()
{
	try
	{
		const unsigned short port_num = 13;

		boost::asio::io_service ioService;
		boost::asio::ip::udp::socket socket(ioService, boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), port_num));

		for (;;)
		{
			boost::array<char, 1> recv_buf;
			boost::asio::ip::udp::endpoint remote_endpoint;
			boost::system::error_code error;
			socket.receive_from(boost::asio::buffer(recv_buf), remote_endpoint, 0, error);
			if (error && error != boost::asio::error::message_size)
				throw boost::system::system_error(error);
			
			const std::string message = make_daytime_string();

			boost::system::error_code ignored_error;
			socket.send_to(boost::asio::buffer(message), remote_endpoint, 0, ignored_error);
		}
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

void asio_udp_server()
{
	local::asio_async_udp_server();
	//local::asio_sync_udp_server();
}
