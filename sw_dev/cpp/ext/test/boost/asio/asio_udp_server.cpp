#if defined(WIN32) || defined(_WIN32)
#define _WIN32_WINNT_NT4 0x0400  // Windows NT 4.0
#define _WIN32_WINNT_WIN2K 0x0500  // Windows 2000
#define _WIN32_WINNT_WINXP 0x0501  // Windows XP
#define _WIN32_WINNT_WIN7 0x0601  // Windows 7
#define _WIN32_WINNT_WIN10 0x0A00  // Windows 10
#define _WIN32_WINNT _WIN32_WINNT_WIN7
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
	enum { port_num = 30001 };

public:
	udp_server(boost::asio::io_service &io_service)
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

	void handle_receive(const boost::system::error_code &error, std::size_t bytes_transferred)
	{
		if (!error || error == boost::asio::error::message_size)
		{
            // TODO [delete] >> for display.
            boost::array<char, 10>::iterator itEnd = recv_buffer_.begin();
            std::advance(itEnd, bytes_transferred);
            std::cout << std::string(recv_buffer_.begin(), itEnd) << std::endl;
            //std::cout.write(recv_buffer_.data(), (std::streamsize)bytes_transferred);
            //std::cout << std::endl;

            // TODO [delete] >> echo server.
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
	boost::array<char, 10> recv_buffer_;
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
		std::cerr << "Boost.Asio exception: " << e.what() << std::endl;
	}
}

void asio_sync_udp_server()
{
	try
	{
		const unsigned short port_num = 30001;

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
		std::cerr << "Boost.Asio exception: " << e.what () << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

void asio_udp_server()
{
	local::asio_async_udp_server();
	//local::asio_sync_udp_server();
}
