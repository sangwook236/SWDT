#if defined(WIN32) || defined(_WIN32)
#define _WIN32_WINNT_NT4 0x0400  // Windows NT 4.0
#define _WIN32_WINNT_WIN2K 0x0500  // Windows 2000
#define _WIN32_WINNT_WINXP 0x0501  // Windows XP
#define _WIN32_WINNT_WIN7 0x0601  // Windows 7
#define _WIN32_WINNT_WIN10 0x0A00  // Windows 10
#define _WIN32_WINNT _WIN32_WINNT_WIN7
#endif

#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <iostream>


namespace {
namespace local {

void asio_async_tcp_client()
{
	boost::asio::io_service ioService;
}

void asio_sync_tcp_client()
{
	try
	{
		boost::asio::io_service ioService;
		boost::asio::ip::tcp::resolver resolver(ioService);
		//boost::asio::ip::tcp::resolver::query query("127.0.0.1", "daytime");  // use a service name.
		boost::asio::ip::tcp::resolver::query query("127.0.0.1", "30001");  // use a port number.
		boost::asio::ip::tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
		boost::asio::ip::tcp::resolver::iterator itEnd;
		boost::asio::ip::tcp::socket socket(ioService);

		boost::system::error_code error = boost::asio::error::host_not_found;
		while (error && endpoint_iterator != itEnd)
		{
			socket.close();
			socket.connect(*endpoint_iterator++, error);
		}
		if (error)
			throw boost::system::system_error(error);

		for (;;)
		{
			boost::array<char, 128> buf;
			boost::system::error_code error;

			{
				const std::size_t len = socket.read_some(boost::asio::buffer(buf), error);
				if (boost::asio::error::eof == error)
					break;  // Connection closed cleanly by peer.
				else if (error)
					throw boost::system::system_error(error);  // Some other error.

				std::cout.write(buf.data(), (std::streamsize)len);
			}

			{
				const std::string msg("!@#$% a TCP test message ^&*()");

				buf.assign('\0');
				std::copy(msg.begin(), msg.end(), buf.begin());

				const std::size_t len = socket.write_some(boost::asio::buffer(buf), error);
				if (boost::asio::error::eof == error)
					break;  // Connection closed cleanly by peer.
				else if (error)
					throw boost::system::system_error(error);  // Some other error.

				//std::cout.write(buf.data(), (std::streamsize)len);
			}

			{
				buf.assign('\0');
				const size_t len = socket.read_some(boost::asio::buffer(buf), error);
				if (boost::asio::error::eof == error)
					break;  // Connection closed cleanly by peer.
				else if (error)
					throw boost::system::system_error(error);  // Some other error.

				std::cout.write(buf.data(), (std::streamsize)len);
			}
		}
	}
	catch (const std::exception &e)
	{
		std::cerr << "Boost.Asio exception: " << e.what() << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

void asio_tcp_client()
{
	//local::asio_async_tcp_client();
	local::asio_sync_tcp_client();
}
