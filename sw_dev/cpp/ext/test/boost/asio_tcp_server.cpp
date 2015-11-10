#if defined(WIN32)
#define _WIN32_WINNT_NT4 0x0400  // Windows NT 4.0
#define _WIN32_WINNT_WIN2K 0x0500  // Windows 2000
#define _WIN32_WINNT_WINXP 0x0501  // Windows XP
#define _WIN32_WINNT_WIN7 0x0601  // Windows 7
#define _WIN32_WINNT_WIN10 0x0A00  // Windows 10
#define _WIN32_WINNT _WIN32_WINNT_WIN7
#endif

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

// simulation of a third party library that wants to perform read and write
// operations directly on a socket. It needs to be polled to determine whether
// it requires a read or write operation, and notified when the socket is ready
// for reading or writing.
class tcp_session
{
public:
	tcp_session(boost::asio::ip::tcp::socket &socket)
	: socket_(socket), state_(writing)
	{
        // TODO [delete] >> echo server.
		const std::string message = make_daytime_string();
		write_buffer_ = boost::asio::buffer(message.c_str(), message.length());

		state_ = boost::asio::buffer_size(write_buffer_) > 0 ? writing : reading;
	}
	~tcp_session()
	{
	}

	// returns true if the third party library wants to be notified when the socket is ready for reading.
	bool want_read() const
	{
		return state_ == reading;
	}

	// notify that third party library that it should perform its read operation.
	void do_read(boost::system::error_code &ec)
	{
		if (const std::size_t len = socket_.read_some(boost::asio::buffer(data_), ec))
		{
            // TODO [delete] >> for display.
            const std::string msg(data_.begin(), data_.end());
            std::cout << "received data of length " << len << " : " << msg << std::endl;

			write_buffer_ = boost::asio::buffer(data_, len);
			state_ = boost::asio::buffer_size(write_buffer_) > 0 ? writing : reading;
		}
	}

	// returns true if the third party library wants to be notified when the socket is ready for writing.
	bool want_write() const
	{
		return state_ == writing;
	}

	// notify that third party library that it should perform its write operation.
	void do_write(boost::system::error_code &ec)
	{
		if (boost::asio::buffer_size(write_buffer_) > 0)
		{
			if (std::size_t len = socket_.write_some(boost::asio::buffer(write_buffer_), ec))
			{
				write_buffer_ = write_buffer_ + len;
				state_ = boost::asio::buffer_size(write_buffer_) > 0 ? writing : reading;
			}
		}
		else state_ = reading;
	}

private:
	boost::asio::ip::tcp::socket &socket_;
	enum { reading, writing } state_;

	boost::array<char, 128> data_;
	boost::asio::const_buffer write_buffer_;
};

class tcp_connection: public boost::enable_shared_from_this<tcp_connection>
{
public:
	typedef boost::shared_ptr<tcp_connection> pointer;

private:
	tcp_connection(boost::asio::io_service &io_service)
	: socket_(io_service), session_impl_(socket_), read_in_progress_(false), write_in_progress_(false)
	{
	}

public:
	static pointer create(boost::asio::io_service &io_service)
	{
		return pointer(new tcp_connection(io_service));
	}

	boost::asio::ip::tcp::socket & socket()
	{
		return socket_;
	}

	void start()
	{
		// put the socket into non-blocking mode.
		boost::asio::ip::tcp::socket::non_blocking_io non_blocking_io(true);
		socket_.io_control(non_blocking_io);

		start_operations();
	}

private:
	void start_operations()
	{
		// start a read operation if the third party library wants one.
		if (session_impl_.want_read() && !read_in_progress_)
		{
			read_in_progress_ = true;
			socket_.async_read_some(
				boost::asio::null_buffers(),
				boost::bind(&tcp_connection::handle_read, shared_from_this(), boost::asio::placeholders::error)
			);
		}

		// start a write operation if the third party library wants one.
		if (session_impl_.want_write() && !write_in_progress_)
		{
			write_in_progress_ = true;
			socket_.async_write_some(
				boost::asio::null_buffers(),
				boost::bind(&tcp_connection::handle_write, shared_from_this(), boost::asio::placeholders::error)
			);
		}
	}

	void handle_read(boost::system::error_code ec)
	{
		read_in_progress_ = false;

		// notify third party library that it can perform a read.
		if (!ec)
			session_impl_.do_read(ec);

		// the third party library successfully performed a read on the socket.
		// start new read or write operations based on what it now wants.
		if (!ec || ec == boost::asio::error::would_block)
			start_operations();
		// otherwise, an error occurred. Closing the socket cancels any outstanding
		// asynchronous read or write operations. The tcp_connection object will be
		// destroyed automatically once those outstanding operations complete.
		else
			socket_.close();
	}

	void handle_write(boost::system::error_code ec)
	{
		write_in_progress_ = false;

		// notify third party library that it can perform a write.
		if (!ec)
			session_impl_.do_write(ec);

		// the third party library successfully performed a write on the socket.
		// start new read or write operations based on what it now wants.
		if (!ec || ec == boost::asio::error::would_block)
			start_operations();
		// otherwise, an error occurred. Closing the socket cancels any outstanding
		// asynchronous read or write operations. The tcp_connection object will be
		// destroyed automatically once those outstanding operations complete.
		else
			socket_.close();
	}

private:
	boost::asio::ip::tcp::socket socket_;
	tcp_session session_impl_;
	bool read_in_progress_;
	bool write_in_progress_;
};

class tcp_server
{
public:
	static const unsigned short port_num = 30001;

public:
	tcp_server(boost::asio::io_service &io_service)
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

	void handle_accept(tcp_connection::pointer new_connection, const boost::system::error_code &ec)
	{
		if (!ec)
		{
			new_connection->start();
			start_accept();
		}
	}

private:
	boost::asio::ip::tcp::acceptor acceptor_;
};

void asio_async_tcp_server()
{
	try
	{
		boost::asio::io_service ioService;
		tcp_server server(ioService);

		ioService.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << "Boost.Asio exception: " << e.what() << std::endl;
	}
}

void asio_sync_tcp_server()
{
	try
	{
		const unsigned short port_num = 30001;

		boost::asio::io_service ioService;
		boost::asio::ip::tcp::acceptor acceptor(ioService, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port_num));

		while (true)
		{
			boost::asio::ip::tcp::socket socket(ioService);
			acceptor.accept(socket);

			const std::string message = make_daytime_string();

			boost::system::error_code ignored_error;
			boost::asio::write(socket, boost::asio::buffer(message), boost::asio::transfer_all(), ignored_error);
		}
	}
	catch (std::exception &e)
	{
		std::cerr << "Boost.Asio exception: " << e.what() << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

void asio_tcp_server()
{
	local::asio_async_tcp_server();
	//local::asio_sync_tcp_server();
}
