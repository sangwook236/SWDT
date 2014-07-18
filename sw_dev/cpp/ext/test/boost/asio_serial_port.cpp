#if defined(WIN32)
#define _WIN32_WINNT 0x0501
#endif

#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/smart_ptr.hpp>
#include <deque>
#include <iostream>


#if !defined(BOOST_ASIO_HAS_SERIAL_PORT)
#error Boost.Asio does not support serial port
#endif


namespace {
namespace local {

class simple_serial_port_handler
{
public:
	simple_serial_port_handler(boost::asio::io_service &io_service)
	: io_service_(io_service), port_(io_service), is_active_(false)
	{}
	~simple_serial_port_handler()
	{
		if (is_active_)
			close();
	}

public:
	bool open(const std::string &port_name, const unsigned int baud_rate)
	{
		port_.open(port_name);
		if (port_.is_open())
		{
			is_active_ = true;

			// options must be set after port was opened
			port_.set_option(boost::asio::serial_port::baud_rate(baud_rate));
			port_.set_option(boost::asio::serial_port::flow_control(boost::asio::serial_port::flow_control::none));
			port_.set_option(boost::asio::serial_port::parity(boost::asio::serial_port::parity::none));
			port_.set_option(boost::asio::serial_port::stop_bits(boost::asio::serial_port::stop_bits::one));
			port_.set_option(boost::asio::serial_port::character_size(8));

			//
			boost::array<char, max_read_length> buf;

			port_.async_read_some(
				boost::asio::buffer(buf),
				boost::bind(&simple_serial_port_handler::handle_read_header, this, boost::asio::placeholders::error, max_read_length)
			);

			return true;
		}
		else
		{
			std::cerr << "failed to open serial port" << std::endl;
			return false;
		}
	}

	void close()
	{
		io_service_.post(boost::bind(&simple_serial_port_handler::do_close, this));
	}

	bool is_active() const  // return true if the serial port is still active
	{
		return is_active_;
	}

	void write(const std::string &msg)
	{
		io_service_.post(boost::bind(&simple_serial_port_handler::do_write, this, msg));
	}

private:
	void handle_read_header(const boost::system::error_code &error, const std::size_t len)
	{
		if (!error)
		{
			boost::array<char, max_read_length> buf;

			port_.async_read_some(
				boost::asio::buffer(buf),
				boost::bind(&simple_serial_port_handler::handle_read_body, this, boost::asio::placeholders::error, len)
			);

			std::cout.write(buf.data(), (std::streamsize)len);
		}
		else
		{
			do_close();
		}
	}

	void handle_read_body(const boost::system::error_code &error, const std::size_t len)
	{
		if (!error)
		{
			boost::array<char, max_read_length> buf;

			port_.async_read_some(
				boost::asio::buffer(buf),
				boost::bind(&simple_serial_port_handler::handle_read_header, this, boost::asio::placeholders::error, len)
			);

			std::cout.write(buf.data(), (std::streamsize)len);
		}
		else
		{
			do_close();
		}
	}

	void do_write(const std::string &msg)
	{
		msg_ = msg;

		port_.async_write_some(
			boost::asio::buffer(msg_),
			boost::bind(&simple_serial_port_handler::handle_write, this, boost::asio::placeholders::error, msg_.length())
		);
	}

	void handle_write(const boost::system::error_code &error, const std::size_t len)
	{
		if (!error)
		{
#if 1
			// for test.
			std::cout << "sent: " << msg_.substr(0, len) << std::endl;
#endif

			msg_.erase(0, len);

			if (!msg_.empty())
			{
				port_.async_write_some(
					boost::asio::buffer(msg_),
					boost::bind(&simple_serial_port_handler::handle_write, this, boost::asio::placeholders::error, 10)
				);
			}
		}
		else
		{
			do_close();
		}
	}

	void do_close()
	{
		port_.close();
		is_active_ = false;
	}

private:
	static const size_t max_read_length = 128;

	boost::asio::io_service &io_service_;
	boost::asio::serial_port port_;

	bool is_active_;

	std::string msg_;
};

class better_serial_port_handler
{
public:
	better_serial_port_handler(boost::asio::io_service &io_service)
	: io_service_(io_service), port_(io_service), is_active_(false),
	  //read_msgs_(), write_msgs_()
	  read_buf_(8192), write_msgs_()
	{}
	~better_serial_port_handler()
	{
		if (is_active_)
			close();
	}

public:
	bool open(const std::string &port_name, const unsigned int baud_rate)
	{
		port_.open(port_name);
		if (port_.is_open())
		{
			is_active_ = true;

			// options must be set after port was opened
			port_.set_option(boost::asio::serial_port::baud_rate(baud_rate));
			port_.set_option(boost::asio::serial_port::flow_control(boost::asio::serial_port::flow_control::none));
			port_.set_option(boost::asio::serial_port::parity(boost::asio::serial_port::parity::none));
			port_.set_option(boost::asio::serial_port::stop_bits(boost::asio::serial_port::stop_bits::one));
			port_.set_option(boost::asio::serial_port::character_size(8));

			read_start();

			return true;
		}
		else
		{
			std::cerr << "failed to open serial port" << std::endl;
			return false;
		}
	}

	void close()
	{
		io_service_.post(boost::bind(&better_serial_port_handler::do_close, this, boost::system::error_code()));
	}

	void cancel()
	{
		io_service_.post(boost::bind(&better_serial_port_handler::do_cancel, this, boost::system::error_code()));
	}

	bool is_active() const  // return true if the serial port is still active
	{
		return is_active_;
	}

	void write(const std::string &msg)
	{
		io_service_.post(boost::bind(&better_serial_port_handler::do_write, this, msg));
	}

private:
	void read_start()
	{
		port_.async_read_some(
			boost::asio::buffer(read_msg_, max_read_length),
			boost::bind(&better_serial_port_handler::read_complete, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred)
		);
	}

	void read_complete(const boost::system::error_code &error, size_t bytes_transferred)
	{
		if (!error)
		{
			// 1. push received data.
			//read_msgs_.push_back(read_msg_);
			for (size_t i = 0; i < bytes_transferred; ++i)
				read_buf_.push_back(read_msg_[i]);

			// 2. process buffered data.

			// 3. pop processed data.
			//read_msgs_.pop_front();

#if 1
			// for test.
			std::cout << "***** read: " << std::endl;
			std::cout.write(read_msg_, (std::streamsize)bytes_transferred);
			std::cout << std::endl;
#endif

			read_start();
		}
		else
			do_close(error);
	}

	void do_write(const std::string &msg)
	{
		const bool write_in_progress = !write_msgs_.empty();
		write_msgs_.push_back(msg);
		if (!write_in_progress)
			write_start();
	}

	void write_start()
	{
		boost::asio::async_write(
			port_,
			boost::asio::buffer(write_msgs_.front(), write_msgs_.front().length()),
			boost::bind(&better_serial_port_handler::write_complete, this, boost::asio::placeholders::error)
		);
	}

	void write_complete(const boost::system::error_code &error)
	{
		if (!error)
		{
#if 1
			// for test.
			std::cout << "***** write: " << std::endl;
			std::cout << write_msgs_.front().c_str() << std::endl;
#endif

			write_msgs_.pop_front();
			if (!write_msgs_.empty())
				write_start();
		}
		else
			do_close(error);
	}

	void do_close(const boost::system::error_code &error)
	{
		if (error == boost::asio::error::operation_aborted)  // if this call is the result of a timer cancel()
			return;

		if (error == boost::asio::error::eof)
		{
		}
		else if (error)
			std::cerr << "error: " << error.message() << std::endl;

		port_.close();
		is_active_ = false;
	}

	void do_cancel(const boost::system::error_code &error)
	{
		if (error == boost::asio::error::eof)
		{
		}
		else if (error)
			std::cerr << "error: " << error.message() << std::endl;

		port_.cancel();
	}

private:
	static const size_t max_read_length = 512;

	boost::asio::io_service &io_service_;
	boost::asio::serial_port port_;

	bool is_active_;

	char read_msg_[max_read_length];
	//std::deque<std::string> read_msgs_;
	boost::circular_buffer<char> read_buf_;
	std::deque<std::string> write_msgs_;
};

struct serial_port_thread_functor
{
	serial_port_thread_functor(boost::asio::io_service &ioService)
	: ioService_(ioService)
	{}
	~serial_port_thread_functor()
	{}

public:
	void operator()()
	{
		std::cout << "serial port thread is started" << std::endl;
		ioService_.run();
		std::cout << "serial port thread is terminated" << std::endl;
	}

private:
	boost::asio::io_service &ioService_;
};

void asio_async_serial_port_simple()
{
	try
	{
		boost::asio::io_service ioService;
		//boost::asio::serial_port_service serialService(ioService);
		simple_serial_port_handler handler(ioService);
		if (!handler.open("COM1", 57600))
		{
			std::cout << "serial port fails to be opened" << std::endl;
			return;
		}

#if 1
		// for test.
		handler.write("serial port: test message 1");
		//handler.write("serial port: test message 2");  // Oops!!! error
#endif

		ioService.run();

		std::cout << "io_service is terminated" << std::endl;
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}

void asio_async_serial_port_better()
{
	try
	{
		const int mode = 2;

		// one io service & one serial port (no thread)
		if (0 == mode)
		{
			boost::asio::io_service ioService;
			//boost::asio::serial_port_service serialService(ioService);
			better_serial_port_handler handler(ioService);
			if (!handler.open("COM1", 57600))
			{
				std::cout << "serial port fails to be opened" << std::endl;
				return;
			}

#if 1
			// for test.
			handler.write("serial port: test message 1");
			handler.write("serial port: test message 2");
			//handler.cancel();
#endif

			ioService.run();

			std::cout << "io_service is terminated" << std::endl;
		}
		// one io service & two serial ports (one thread)
		else if (1 == mode)
		{
			boost::asio::io_service ioService;
			better_serial_port_handler handler1(ioService);
			better_serial_port_handler handler2(ioService);
			if (!handler1.open("COM1", 57600) || !handler2.open("COM10", 57600))
			{
				std::cout << "serial ports fail to be opened" << std::endl;
				return;
			}

			boost::scoped_ptr<boost::thread> thrd(new boost::thread(serial_port_thread_functor(ioService)));
#if defined(WIN32)
			Sleep(0);  // un-necessary
#else
            boost::this_thread::yield();
#endif

#if 1
			// for test.
			handler1.write("serial port 1: test message #1");
			handler2.write("serial port 2: test message #1");
			handler1.write("serial port 1: test message #2");
			handler2.write("serial port 2: test message #2");
#endif

			std::cout << "wait for joining thread" << std::endl;
			if (thrd.get()) thrd->join();
		}
		// two io services & two serial ports (two threads)
		else if (2 == mode)
		{
			boost::asio::io_service ioService1, ioService2;
			better_serial_port_handler handler1(ioService1);
			better_serial_port_handler handler2(ioService2);
			if (!handler1.open("COM1", 57600) || !handler2.open("COM10", 57600))
			{
				std::cout << "serial ports fail to be opened" << std::endl;
				return;
			}

			boost::scoped_ptr<boost::thread> thrd1(new boost::thread(serial_port_thread_functor(ioService1)));
			boost::scoped_ptr<boost::thread> thrd2(new boost::thread(serial_port_thread_functor(ioService2)));
#if defined(WIN32)
			Sleep(0);  // un-necessary
#else
            boost::this_thread::yield();
#endif

#if 1
			// for test.
			handler1.write("serial port 1: test message #1");
			handler2.write("serial port 2: test message #1");
			handler1.write("serial port 1: test message #2");
			handler2.write("serial port 2: test message #2");
#endif

			std::cout << "wait for joining thread" << std::endl;
			if (thrd1.get()) thrd1->join();
			if (thrd2.get()) thrd2->join();
		}
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}

void asio_sync_serial_port()
{
	try
	{
		boost::asio::io_service ioService;
		//boost::asio::serial_port_service serialService(ioService);

		const std::string port_name("COM1");
		//boost::asio::serial_port port(ioService, port_name);
		boost::asio::serial_port port(ioService);

		port.open(port_name);
		if (port.is_open())
		{
			// options must be set after port was opened
			port.set_option(boost::asio::serial_port::baud_rate(59200));
			port.set_option(boost::asio::serial_port::flow_control(boost::asio::serial_port::flow_control::none));
			port.set_option(boost::asio::serial_port::parity(boost::asio::serial_port::parity::none));
			port.set_option(boost::asio::serial_port::stop_bits(boost::asio::serial_port::stop_bits::one));
			port.set_option(boost::asio::serial_port::character_size(8));

			const std::string message("serial port: test message");
			port.write_some(boost::asio::buffer(message));

			for (;;)
			{
				const size_t max_read_length = 128;
				boost::array<char, max_read_length> buf;
				boost::system::error_code error;

				const std::size_t len = port.read_some(boost::asio::buffer(buf), error);
				if (error == boost::asio::error::eof)
					break; // connection closed cleanly by peer.
				else if (error)
					throw boost::system::system_error(error); // some other error.

				std::cout.write(buf.data(), (std::streamsize)len);
			}

			port.close();
		}
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

void asio_serial_port()
{
	//local::asio_async_serial_port_simple();
	local::asio_async_serial_port_better();
	//local::asio_sync_serial_port();
}
