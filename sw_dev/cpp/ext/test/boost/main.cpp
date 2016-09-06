#if defined(WIN32) || defined(_WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
	void utility();
	void utility_binary();

	bool program_options(int argc, char *argv[]);
	void filesystem();
	void conversion();
	void type_traits();

	void date_time();
	void chrono();
	void progress_timer();
	void cpu_timer();

	void factory();
	void smart_ptr();
	void any();
	void array_main();
	void tuple();
	void multi_array();
	void multi_index();
	void property_map();
	void property_tree();

	void function();
	void bind();
	void signals_slots();

	void serialization();

	void tokenizer();
	void string_algorithms();
	void spirit_classic();
	void statechart();
	void thread();
	void interprocess();

	void units();
	void math_constants();
	void math_floating_point_utilities();
	void math_special_functions();
	void math_statistical_distributions();
	void random_boost();
	void ublas();
	void math_bindings();
	void numeric_bindings();
	void multiprecision();

	void polygon();
	void geometry();
	void graph();
	void graph_parallel();
	void ordinary_differential_equation();

	void asio_timer();
	void asio_synchronizing_handler();
	void asio_line_based_operation();
	void asio_tcp_server();
	void asio_tcp_client();
	void asio_udp_server();
	void asio_udp_client();
	void asio_tcp_udp_server();
	void asio_multicast_sender();
	void asio_multicast_receiver();
	void asio_udp_broadcast();
	void asio_serial_port();

	void image();

	int retval = EXIT_SUCCESS;
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
			//chrono();  // Not yet implemented.
			//progress_timer();  // deprecated.
			//cpu_timer();
		}

		{
			//factory();

			//smart_ptr();
			//any();
			//array_main();
			//tuple();
			//multi_array();
			//multi_index();

			//property_map();
			//property_tree();
		}

		{
			//function();
			//bind();
			//signals_slots();

			//serialization();
		}

		{
			//tokenizer();
			//string_algorithms();

			//spirit_classic();
			//statechart();
			//thread();
			//interprocess();  // Not yet implemented.
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

			//multiprecision();  // Not yet implemented.

			//polygon();
			//geometry();

			graph();
			//graph_parallel();  // Not yet implemented.

			//ordinary_differential_equation();  // Not yet implemented.
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

            //asio_multicast_sender();
            //asio_multicast_receiver();
            //asio_udp_broadcast();
		}

		{
			//image();
		}
	}
    catch (const std::bad_alloc &e)
	{
		std::cout << "std::bad_alloc caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cout << "unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
