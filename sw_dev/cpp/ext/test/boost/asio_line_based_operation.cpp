#if defined(WIN32)
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

}  // namespace local
}  // unnamed namespace

void asio_line_based_operation()
{
	const std::string server = "www.boost.org";
	const std::string path = "/LICENSE_1_0.txt";

	boost::asio::io_service io_service;

	// Get a list of endpoints corresponding to the server name.
	boost::asio::ip::tcp::resolver resolver(io_service);
	boost::asio::ip::tcp::resolver::query query(server, "http");
	boost::asio::ip::tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
	boost::asio::ip::tcp::resolver::iterator end;

	// Try each endpoint until we successfully establish a connection.
	boost::asio::ip::tcp::socket socket(io_service);
	boost::system::error_code error = boost::asio::error::host_not_found;
	while (error && endpoint_iterator != end)
	{
		socket.close();
		socket.connect(*endpoint_iterator++, error);
	}
	if (error)
		throw boost::system::system_error(error);

	// Form the request. We specify the "Connection: close" header so that the
	// server will close the socket after transmitting the response. This will
	// allow us to treat all data up until the EOF as the content.
	boost::asio::streambuf request;
	std::ostream request_stream(&request);
	request_stream << "GET " << path << " HTTP/1.0\r\n";
	request_stream << "Host: " << server << "\r\n";
	request_stream << "Accept: */*\r\n";
	request_stream << "Connection: close\r\n\r\n";

	// Send the request.
	boost::asio::write(socket, request);

	// Read the response status line.
	boost::asio::streambuf response;
	boost::asio::read_until(socket, response, "\r\n");  // line-based operation

	// Check that response is OK.
	std::istream response_stream(&response);
	std::string http_version;
	response_stream >> http_version;
	unsigned int status_code;
	response_stream >> status_code;
	std::string status_message;
	std::getline(response_stream, status_message);
	if (!response_stream || http_version.substr(0, 5) != "HTTP/")
	{
		std::cout << "Invalid response\n";
		return;
	}
	if (status_code != 200)
	{
		std::cout << "Response returned with status code " << status_code << "\n";
		return;
	}

	// Read the response headers, which are terminated by a blank line.
	boost::asio::read_until(socket, response, "\r\n\r\n");

	// Process the response headers.
	std::string header;
	while (std::getline(response_stream, header) && header != "\r")
		std::cout << header << "\n";
	std::cout << "\n";

	// Write whatever content we already have to output.
	if (response.size() > 0)
		std::cout << &response;

	// Read until EOF, writing data to output as we go.
	while (boost::asio::read(socket, response,
		boost::asio::transfer_at_least(1), error))
		std::cout << &response;
	if (error != boost::asio::error::eof)
		throw boost::system::system_error(error);
}
