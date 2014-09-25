#include "../open_source_ahrs_lib/MadgwickAHRS.h"
#include "../open_source_ahrs_lib/MahonyAHRS.h"
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


namespace {
namespace local {

struct imu_data_type
{
public:
	imu_data_type()
	: accelx(0.0), accely(0.0), accelz(0.0), gyrox(0.0), gyroy(0.0), gyroz(0.0), magx(0.0), magy(0.0), magz(0.0)
	{}
	imu_data_type(const imu_data_type &rhs)
	: accelx(rhs.accelx), accely(rhs.accely), accelz(rhs.accelz), gyrox(rhs.gyrox), gyroy(rhs.gyroy), gyroz(rhs.gyroz), magx(rhs.magx), magy(rhs.magy), magz(rhs.magz)
	{}

private:
	//imu_data_type & operator=(const imu_data_type &rhs) const;  // not implemented.

public:
	imu_data_type operator+(const imu_data_type &rhs) const
	{
		imu_data_type result;
		result.accelx = accelx + rhs.accelx;
		result.accely = accely + rhs.accely;
		result.accelz = accelz + rhs.accelz;
		result.gyrox = gyrox + rhs.gyrox;
		result.gyroy = gyroy + rhs.gyroy;
		result.gyroz = gyroz + rhs.gyroz;
		result.magx = magx + rhs.magx;
		result.magy = magy + rhs.magy;
		result.magz = magz + rhs.magz;
		return result;
	}

public:
	double accelx, accely, accelz;
	double gyrox, gyroy, gyroz;
	double magx, magy, magz;
};

// [ref] abstract_serial_port_handler class in ${HW_DEV_HOME}/ext/test/imu/invensense/invensense_main.cpp.
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

// [ref] serial_port_handler_for_mpu6050 class in ${HW_DEV_HOME}/ext/test/imu/invensense/invensense_main.cpp.
class serial_port_handler_for_mpu6050 : public abstract_serial_port_handler
{
public:
	typedef abstract_serial_port_handler base_type;

public:
	serial_port_handler_for_mpu6050(boost::asio::io_service &io_service)
	: base_type(io_service),
	  imu_data_list_(20)
	{
		// FIXME [delete] >> for test.
		outStream_.open("./data/ahrs/mpu6050_open_source_ahrs.dat");
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
	 	const double M_PI = std::atan(1.0) * 4.0;
	 	const double DEG2RAD = M_PI / 180.0;

		//std::cout << packet << std::endl;
		
		// accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z.
		std::vector<short> values;
		values.reserve(6);
		parse_imu_values(packet, values);
		if (values.size() == 6)
		{
			// Sensitivity = -2g to 2g at 16Bit : -2g = -32768 && 2g = 32768.
			const double ACCELEROMETER_SENSITIVITY = 16384.0;  // (2^16 / 2) / 2 = 32768 / 2.
			// Sensitivity = -500 [deg/sec] to 500 [deg/sec] at 16Bit : -500 [deg/sec] = -32768 && 500 [deg/sec] = 32767.
			//const double GYROSCOPE_SENSITIVITY = 65.536;
			// Sensitivity = -250 [deg/sec] to 250 [deg/sec] at 16Bit : -250 [deg/sec] = -32768 && 250 [deg/sec] = 32767.
			const double GYROSCOPE_SENSITIVITY = 131.072;  // (2^16 / 2) / 250 = 32768 / 250.
 			const double dt = 0.01;  // sampling time: 10 [msec].

			imu_data_type imu_data;
#if 1
			imu_data.accelx = values[0] / ACCELEROMETER_SENSITIVITY;
			imu_data.accely = values[1] / ACCELEROMETER_SENSITIVITY;
			imu_data.accelz = values[2] / ACCELEROMETER_SENSITIVITY;
			imu_data.gyrox = values[3] * DEG2RAD / GYROSCOPE_SENSITIVITY;  // [rad/sec].
			imu_data.gyroy = values[4] * DEG2RAD / GYROSCOPE_SENSITIVITY;  // [rad/sec].
			imu_data.gyroz = values[5] * DEG2RAD / GYROSCOPE_SENSITIVITY;  // [rad/sec].
			imu_data.magx = 0.0;
			imu_data.magy = 0.0;
			imu_data.magz = 0.0;
#else
			imu_data.accelx = values[2] / ACCELEROMETER_SENSITIVITY;
			imu_data.accely = values[1] / ACCELEROMETER_SENSITIVITY;
			imu_data.accelz = values[0] / ACCELEROMETER_SENSITIVITY;
			imu_data.gyrox = values[5] * DEG2RAD / GYROSCOPE_SENSITIVITY;  // [rad/sec].
			imu_data.gyroy = values[4] * DEG2RAD / GYROSCOPE_SENSITIVITY;  // [rad/sec].
			imu_data.gyroz = values[3] * DEG2RAD / GYROSCOPE_SENSITIVITY;  // [rad/sec].
			imu_data.magx = 0.0;
			imu_data.magy = 0.0;
			imu_data.magz = 0.0;
#endif

			imu_data_type imu_data_final;
#if 1
			// Use raw sensor data.

			imu_data_final = imu_data;
#else
			// Use the moving average of raw sensor data.

			imu_data_list_.push_back(imu_data);

			//if (imu_data_list_.full())
			{
				const std::size_t num_imu_data = imu_data_list_.size();

				// Moving average.
				imu_data_final = std::accumulate(imu_data_list_.begin(), imu_data_list_.end(), imu_data_type());
				imu_data_final.accelx /= (double)num_imu_data;
				imu_data_final.accely /= (double)num_imu_data;
				imu_data_final.accelz /= (double)num_imu_data;
				imu_data_final.gyrox /= (double)num_imu_data;
				imu_data_final.gyroy /= (double)num_imu_data;
				imu_data_final.gyroz /= (double)num_imu_data;
				imu_data_final.magx /= (double)num_imu_data;
				imu_data_final.magy /= (double)num_imu_data;
				imu_data_final.magz /= (double)num_imu_data;
			}
#endif

			// FIXME [delete] >> for test.
			outStream_ << imu_data_final.accelx << ',' << imu_data_final.accely << ',' << imu_data_final.accelz << ','
				<< imu_data_final.gyrox << ',' << imu_data_final.gyroy << ',' << imu_data_final.gyroz << std::endl;

			// Filter by AHRS.
#if 0
			// Caution !!!
			// do set sampleFreq in ${PROJECT_HOME}/open_source_ahrs_lib/MadgwickAHRS.cpp.

			// gyroscope : [rad/sec], acceleromter : [g], magnetometer : gauss [G], dt : sampling time.
			//madgwick_ahrs::MadgwickAHRSupdate((float)imu_data_final.gyrox, (float)imu_data_final.gyroy, (float)imu_data_final.gyroz, (float)imu_data_final.accelx, (float)imu_data_final.accely, (float)imu_data_final.accelz, (float)imu_data_final.magx, (float)imu_data_final.magy, (float)imu_data_final.magz);
			madgwick_ahrs::MadgwickAHRSupdateIMU((float)imu_data_final.gyrox, (float)imu_data_final.gyroy, (float)imu_data_final.gyroz, (float)imu_data_final.accelx, (float)imu_data_final.accely, (float)imu_data_final.accelz);

			const float q0 = madgwick_ahrs::q0, q1 = madgwick_ahrs::q1, q2 = madgwick_ahrs::q2, q3 = madgwick_ahrs::q3;
#else
			// Caution !!!
			// do set sampleFreq in ${PROJECT_HOME}/open_source_ahrs_lib/MahonyAHRS.cpp.

			// gyroscope : [rad/sec], acceleromter : [g], magnetometer : gauss [G], dt : sampling time.
			//mahony_ahrs::MahonyAHRSupdate((float)imu_data_final.gyrox, (float)imu_data_final.gyroy, (float)imu_data_final.gyroz, (float)imu_data_final.accelx, (float)imu_data_final.accely, (float)imu_data_final.accelz, (float)imu_data_final.magx, (float)imu_data_final.magy, (float)imu_data_final.magz);
			mahony_ahrs::MahonyAHRSupdateIMU((float)imu_data_final.gyrox, (float)imu_data_final.gyroy, (float)imu_data_final.gyroz, (float)imu_data_final.accelx, (float)imu_data_final.accely, (float)imu_data_final.accelz);

			const float q0 = mahony_ahrs::q0, q1 = mahony_ahrs::q1, q2 = mahony_ahrs::q2, q3 = mahony_ahrs::q3;
#endif

			// Calculate equivalent angle-axis.
			const double theta = 2.0 * std::atan2(std::sqrt(q1*q1 + q2*q2 + q3*q3), q0);
			std::cout << "theta = " << theta;
			const double eps = 1.0e-3;
			if (std::fabs(theta - M_PI) < eps)  // when theta = pi (cos(theta/2) = 0).
				std::cout << ", k = (" << q1 << ',' << q2 << ',' << q3 << ')' << std::endl;
			else if (std::fabs(theta + M_PI) < eps)  // when theta = -pi (cos(theta/2) = 0).
				std::cout << ", k = (" << -q1 << ',' << -q2 << ',' << -q3 << ')' << std::endl;
			else if (std::fabs(theta) < eps || std::fabs(theta + 2.0 * M_PI) < eps || std::fabs(theta - 2.0 * M_PI) < eps)  // when theta = 0, -2 * pi, 2 * pi (sin(theta/2) = 0).
				std::cout << ", k = (" << 0.0 << ',' << 0.0 << ',' << 0.0 << ')' << std::endl;
			else
			{
				const double factor = 1.0 / std::sin(0.5 * theta);
				std::cout << ", k = (" << (q1 * factor) << ',' << (q2 * factor) << ',' << (q3 * factor) << ')' << std::endl;
			}
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
	boost::circular_buffer<imu_data_type> imu_data_list_;

	// FIXME [delete] >> for test.
	std::ofstream outStream_;
};

// [ref] mpu6050_processing_main() in ${HW_DEV_HOME}/ext/test/imu/invensense/invensense_main.cpp.
void filter_by_ahrs_using_mpu6050()
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

namespace my_open_source_ahrs {

}  // namespace my_open_source_ahrs

int open_source_ahrs_main(int argc, char *argv[])
{
	local::filter_by_ahrs_using_mpu6050();

	return 0;
}
