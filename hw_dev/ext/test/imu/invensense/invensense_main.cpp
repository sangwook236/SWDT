#include <iostream>
#include <boost/spirit/include/classic_core.hpp>
#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <boost/bind.hpp>
//#include <boost/thread.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/smart_ptr.hpp>
#include <numeric>
#include <deque>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cassert>


namespace {
namespace local {

// [ref] http://www.pieter-jan.com/node/11.
void complementary_filter_for_mpu6050(const short accData[3], const short gyrData[3], double &pitch, double &roll)
{
	// InvenSense MPU6050.

#if defined(_MSC_VER)
 	const double M_PI = std::atan(1.0) * 4.0;
#endif

	// Sensitivity = -2g to 2g at 16Bit : 2g = 32768 && 0.5g = 8192.
	const double ACCELEROMETER_SENSITIVITY = 8192.0;
	// Sensitivity = -500 [deg/sec] to 500 [deg/sec] at 16Bit : -500 [deg/sec] = -32768 && 500 [deg/sec] = 32767.
	//const double GYROSCOPE_SENSITIVITY = 65.536;
	// Sensitivity = -250 [deg/sec] to 250 [deg/sec] at 16Bit : -250 [deg/sec] = -32768 && 250 [deg/sec] = 32767.
	const double GYROSCOPE_SENSITIVITY = 131.072;
 	const double dt = 0.01;  // sample rate: 10 [msec].

	// Integrate the gyroscope data -> int(angularSpeed) = angle
	pitch += ((double)gyrData[0] / GYROSCOPE_SENSITIVITY) * dt;  // Angle around the X-axis
	roll -= ((double)gyrData[1] / GYROSCOPE_SENSITIVITY) * dt;  // Angle around the Y-axis

	// Compensate for drift with accelerometer data if !bullshit
	const int forceMagnitudeApprox = std::abs(accData[0]) + std::abs(accData[1]) + std::abs(accData[2]);
	if (forceMagnitudeApprox > 8192 && forceMagnitudeApprox < 32768)
	{
		// Turning around the X axis results in a vector on the Y-axis
		const double pitchAcc = std::atan2((double)accData[1], (double)accData[2]) * 180.0 / M_PI;
		pitch = pitch * 0.98 + pitchAcc * 0.02;

		// Turning around the Y axis results in a vector on the X-axis
		const double rollAcc = std::atan2((double)accData[0], (double)accData[2]) * 180.0 / M_PI;
		roll = roll * 0.98 + rollAcc * 0.02;
	}
}

// [ref] better_serial_port_handler class in ${CPP_EXT_HOME}/test/boost/asio_serial_port.cpp.
class abstract_serial_port_handler
{
public:
	abstract_serial_port_handler(boost::asio::io_service &io_service)
	: io_service_(io_service), port_(io_service), is_active_(false),
	  //read_msgs_(), write_msgs_()
	  read_buf_(8192), write_msgs_()
	{}
	virtual ~abstract_serial_port_handler()
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
		io_service_.post(boost::bind(&abstract_serial_port_handler::do_close, this, boost::system::error_code()));
	}

	void cancel()
	{
		io_service_.post(boost::bind(&abstract_serial_port_handler::do_cancel, this, boost::system::error_code()));
	}

	bool is_active() const  // return true if the serial port is still active
	{
		return is_active_;
	}

	void write(const std::string &msg)
	{
		io_service_.post(boost::bind(&abstract_serial_port_handler::do_write, this, msg));
	}

protected:
	virtual void process_buffered_data() = 0;

private:
	void read_start()
	{
		port_.async_read_some(
			boost::asio::buffer(read_msg_, MAX_READ_LENGTH),
			boost::bind(&abstract_serial_port_handler::read_complete, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred)
		);
	}

	void read_complete(const boost::system::error_code &error, size_t bytes_transferred)
	{
		if (!error)
		{
			// 1. push received data.
			//read_msgs_.push_back(read_msg_);
			for (size_t i = 0; i < bytes_transferred; ++i)
				if (read_buf_.full())
					std::cerr << "receive buffer is full" << std::endl;
				else
					read_buf_.push_back(read_msg_[i]);

			// 2. process buffered data.
			process_buffered_data();

			// 3. pop processed data.
			//read_msgs_.pop_front();

#if 0
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
			boost::bind(&abstract_serial_port_handler::write_complete, this, boost::asio::placeholders::error)
		);
	}

	void write_complete(const boost::system::error_code &error)
	{
		if (!error)
		{
#if 0
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

protected:
	//std::deque<std::string> read_msgs_;
	boost::circular_buffer<char> read_buf_;

private:
	static const size_t MAX_READ_LENGTH = 512;

	boost::asio::io_service &io_service_;
	boost::asio::serial_port port_;

	bool is_active_;

	char read_msg_[MAX_READ_LENGTH];
	std::deque<std::string> write_msgs_;
};

class serial_port_handler_for_mpu6050 : public abstract_serial_port_handler
{
public:
	typedef abstract_serial_port_handler base_type;

public:
	serial_port_handler_for_mpu6050(boost::asio::io_service &io_service)
	: base_type(io_service),
	  accelx_(20), accely_(20), accelz_(20), gyrox_(20), gyroy_(20), gyroz_(20)
	{
		// FIXME [delete] >> for test.
		outStream_.open("./data/imu/mpu6050.dat");
		if (!outStream_.is_open())
			std::cerr << "output stream not opened" << std::endl;
	}
	virtual ~serial_port_handler_for_mpu6050()
	{
		// FIXME [delete] >> for test.
		outStream_.close();
	}

protected:
	/*virtual*/ void process_buffered_data()
	{
		while (!read_buf_.empty())
		{
			const std::string &packet = pop_packet();
			if (packet.empty()) break;
			else process_packet(packet);
		}
	}

private:
	std::string pop_packet()
	{
		while ('<' != read_buf_.front())
		{
			read_buf_.pop_front();

			if (read_buf_.empty()) return std::string();
		}

		size_t len = read_buf_.size();
		for (size_t k = 1; k < len; ++k)
		{
			if ('>' == read_buf_[k])
			{
				std::ostringstream strm;
#if 0
				for (size_t i = 0; i <= k; ++i)
				{
					strm << read_buf_.front();
					read_buf_.pop_front();
				}
#else
				read_buf_.pop_front();
				for (size_t i = 1; i < k; ++i)
				{
					strm << read_buf_.front();
					read_buf_.pop_front();
				}
				read_buf_.pop_front();
#endif
				return strm.str();
			}
		}

		return std::string();
	}

	void process_packet(const std::string &packet)
	{
		//std::cout << packet << std::endl;

		// accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z.
		std::vector<short> values;
		values.reserve(6);
		parse_imu_values(packet, values);
		if (values.size() == 6)
		{
#if 0
			std::cout << packet << " : " << values[0] << ',' << values[1] << ',' << values[2] << ',' << values[3] << ',' << values[4] << ',' << values[5] << std::endl;
#elif 1
			double pitch, roll;
			complementary_filter_for_mpu6050(&values[0], &values[3], pitch, roll);

			std::cout << "roll: " << roll << ", pitch: " << pitch << std::endl;
#endif
		}
	}

	bool parse_imu_values(const std::string &packet, std::vector<short> &values)
	{
		return boost::spirit::classic::parse(
			packet.c_str(),
			// begin grammar
			(
				//boost::spirit::classic::alnum_p[boost::spirit::classic::push_back_a(values)] >> *(boost::spirit::classic::alnum_p[boost::spirit::classic::push_back_a(values)])
				//boost::spirit::classic::real_p[boost::spirit::classic::push_back_a(values)] >> *(',' >> boost::spirit::classic::real_p[boost::spirit::classic::push_back_a(values)])
				boost::spirit::classic::int_p[boost::spirit::classic::push_back_a(values)] >> *(',' >> boost::spirit::classic::int_p[boost::spirit::classic::push_back_a(values)])
			),
			// end grammar
			boost::spirit::classic::space_p
		).full;
	}

private:
	boost::circular_buffer<double> accelx_, accely_, accelz_;
	boost::circular_buffer<double> gyrox_, gyroy_, gyroz_;

	// FIXME [delete] >> for test.
	std::ofstream outStream_;
};

// [ref] asio_async_serial_port_better() in ${CPP_EXT_HOME}/test/boost/asio_serial_port.cpp.
void mpu6050_processing_main()
{
	try
	{
		const std::string port = "COM9";
		const unsigned int baud_rate = 38400;
		{
			boost::asio::io_service ioService;
			//boost::asio::serial_port_service serialService(ioService);
			serial_port_handler_for_mpu6050 handler(ioService);
			if (!handler.open(port, baud_rate))
			{
				std::cout << "serial port not opened" << std::endl;
				return;
			}

#if 0
			// for test.
			handler.write("serial port: test message 1");
			handler.write("serial port: test message 2");
			//handler.cancel();
#endif

			ioService.run();

			std::cout << "io_service is terminated" << std::endl;
		}
	}
	catch (const std::exception &e)
	{
		std::cerr << e.what() << std::endl;
	}
}

}  // namespace local
}  // unnamed namespace

namespace my_invensense {

}  // namespace my_invensense

int invensense_main(int argc, char *argv[])
{
	local::mpu6050_processing_main();

	return 0;
}
