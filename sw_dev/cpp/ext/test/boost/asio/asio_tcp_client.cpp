#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

void asio_async_tcp_client()
{
	throw std::runtime_error("Not yet implemented");

	//boost::asio::io_service ioService;
}

void asio_sync_tcp_client()
{
	try
	{
		boost::asio::io_service ioService;
		boost::asio::ip::tcp::resolver resolver(ioService);
		//boost::asio::ip::tcp::resolver::query query("127.0.0.1", "daytime");  // Use a service name.
		boost::asio::ip::tcp::resolver::query query("127.0.0.1", "30001");  // Use a port number.
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
	catch (const std::exception &ex)
	{
		std::cerr << "Boost.Asio exception: " << ex.what() << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

void asio_tcp_client()
{
	//local::asio_async_tcp_client();  // Not yet implemented.
	local::asio_sync_tcp_client();
}
