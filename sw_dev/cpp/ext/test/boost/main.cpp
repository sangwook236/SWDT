#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


int main(int argc, char *argv[])
{
	void utility();
	void utility_binary();

	bool program_options(int argc, char *argv[]);
	void filesystem();
	void conversion();
	void type_traits();
	void date_time();

	void factory();
	void smart_ptr();
	void any();
	void array_();
	void tuple();
	void multi_array();
	void property_map();
	void property_tree();

	void function();
	void bind();
	void signals_slots();

	void serialization();

	void tokenizer();
	void spirit_classic();
	void statechart();
	void thread();

	void units();
	void math_constants();
	void math_floating_point_utilities();
	void math_special_functions();
	void math_statistical_distributions();
	void random_boost();
	void ublas();
	void math_bindings();
	void numeric_bindings();
	void mtl_matrix();

	void polygon();
	void geometry();

	void graph();

	void asio_timer();
	void asio_synchronizing_handler();
	void asio_line_based_operation();
	void asio_tcp_server();
	void asio_tcp_client();
	void asio_udp_server();
	void asio_udp_client();
	void asio_tcp_udp_server();
	void asio_serial_port();

	void image();

	try
	{
		{
			//utility();
			//utility_binary();

			//program_options(argc, argv);
			//filesystem();
			//conversion();
			//type_traits();
			//date_time();
		}

		{
			//factory();

			//smart_ptr();
			//any();
			//array_();
			//tuple();
			//multi_array();

			//property_map();
			property_tree();
		}

		{
			//function();
			//bind();
			//signals_slots();

			//serialization();
		}

		{
			//tokenizer();
			//spirit_classic();
			//statechart();
			//thread();
		}

		{
			//units();

			//math_constants();
			//math_floating_point_utilities();
			//math_special_functions();
			//math_statistical_distributions();
			//random();
			//ublas();
			//math_bindings();
			//numeric_bindings();
			//mtl_matrix();

			//polygon();
			//geometry();

			//graph();
		}

		{
			//asio_timer();
			//asio_synchronizing_handler();
			//asio_line_based_operation();

			//asio_tcp_server();
			//asio_udp_server();
			//asio_tcp_udp_server();

			//asio_tcp_client();
			//asio_udp_client();

			//asio_serial_port();
		}

		{
			//image();
		}
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception occurred: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::cout << "unknown exception occurred" << std::endl;
	}

	std::cout << "press any key to exit ..." << std::flush;
	std::cin.get();

    return 0;
}
