#if defined(WIN32)
#define _WIN32_WINNT 0x0501
#endif

#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <iostream>


void asio_async_tcp_client();
void asio_sync_tcp_client();

void asio_tcp_client()
{
	//asio_async_tcp_client();
	asio_sync_tcp_client();
}

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
		boost::asio::ip::tcp::resolver::query query("127.0.0.1", "daytime");  // use a service name
		//boost::asio::ip::tcp::resolver::query query("127.0.0.1", "13");  // use a port number
		boost::asio::ip::tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
		boost::asio::ip::tcp::resolver::iterator end;
		boost::asio::ip::tcp::socket socket(ioService);

		boost::system::error_code error = boost::asio::error::host_not_found;
		while (error && endpoint_iterator != end)
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
				const size_t len = socket.read_some(boost::asio::buffer(buf), error);
				if (error == boost::asio::error::eof)
					break; // Connection closed cleanly by peer.
				else if (error)
					throw boost::system::system_error(error); // Some other error.

				std::cout.write(buf.data(), (std::streamsize)len);
			}

			{
				const std::string msg("abcdef");
	
				buf.assign('\0');
				std::copy(msg.begin(), msg.end(), buf.begin());

				const size_t len = socket.write_some(boost::asio::buffer(buf), error);
				if (error == boost::asio::error::eof)
					break; // Connection closed cleanly by peer.
				else if (error)
					throw boost::system::system_error(error); // Some other error.

				//std::cout.write(buf.data(), (std::streamsize)len);
			}

			{
				buf.assign('\0');
				const size_t len = socket.read_some(boost::asio::buffer(buf), error);
				if (error == boost::asio::error::eof)
					break; // Connection closed cleanly by peer.
				else if (error)
					throw boost::system::system_error(error); // Some other error.

				std::cout.write(buf.data(), (std::streamsize)len);
			}
		}
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}
