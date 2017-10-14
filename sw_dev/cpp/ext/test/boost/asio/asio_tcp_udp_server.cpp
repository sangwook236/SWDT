#include <boost/asio.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>
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

class tcp_connection: public boost::enable_shared_from_this<tcp_connection>
{
public:
	typedef boost::shared_ptr<tcp_connection> pointer;

private:
	tcp_connection(boost::asio::io_service& io_service)
		: socket_(io_service)
	{
	}

public:
	static pointer create(boost::asio::io_service& io_service)
	{
		return pointer(new tcp_connection(io_service));
	}

	boost::asio::ip::tcp::socket& socket()
	{
		return socket_;
	}

	void start()
	{
		message_ = make_daytime_string();
		boost::asio::async_write(
			socket_,
			boost::asio::buffer(message_),
			boost::bind(&tcp_connection::handle_write, shared_from_this(), boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred)
		);
	}

private:
	void handle_write(const boost::system::error_code& /*error*/, size_t /*bytes_transferred*/)
	{
	}

private:
	boost::asio::ip::tcp::socket socket_;
	std::string message_;
};

class tcp_server
{
public:
	static const unsigned short port_num = 30001;

public:
	tcp_server(boost::asio::io_service& io_service)
	: acceptor_(io_service, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port_num))
	{
		start_accept();
	}

private:
	void start_accept()
	{
		tcp_connection::pointer new_connection = tcp_connection::create(acceptor_.get_io_service());

		acceptor_.async_accept(
			new_connection->socket(),
			boost::bind(&tcp_server::handle_accept, this, new_connection, boost::asio::placeholders::error)
		);
	}

	void handle_accept(tcp_connection::pointer new_connection, const boost::system::error_code& error)
	{
		if (!error)
		{
			new_connection->start();
			start_accept();
		}
	}

private:
	boost::asio::ip::tcp::acceptor acceptor_;
};

class udp_server
{
public:
	static const unsigned short port_num = 30001;

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

void asio_async_tcp_udp_server()
{
	try
	{
		boost::asio::io_service ioService;
		tcp_server server1(ioService);
		udp_server server2(ioService);

		ioService.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << "Boost.Asio exception: " << e.what() << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

void asio_tcp_udp_server()
{
	local::asio_async_tcp_udp_server();
}
