//#define BOOST_ALL_DYN_LINK 1

#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>
#include <stdexcept>
#include <cstdlib>


int main(int argc, char *argv[])
{
	void utility();
	void utility_binary();

	void log();

	bool program_options(int argc, char *argv[]);
	void format();
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
	void circular_buffer();
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
	void meta_state_machine();  // Meta State Machine (MSM), eUML.
	void state_machine_library();  // State Machine Library (SML).

	void thread();
	void process();
	void interprocess();
	void iostreams();

	void units();
	void math_constants();
	void math_floating_point_utilities();
	void math_special_functions();
	void math_statistical_distributions();
	void accumulator();
	void random_boost();
	void ublas();
	void odeint();
	void math_bindings();
	void numeric_bindings();
	void multiprecision();

	void polygon();
	void geometry();
	void graph();
	void graph_parallel();

	void asio_io_service();
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

	void compute();
	void mpi();

	int retval = EXIT_SUCCESS;
	try
	{
		{
			//utility();
			//utility_binary();

			//log();

			//program_options(argc, argv);
			//format();
			//filesystem();
			//conversion();
			//type_traits();

			//date_time();
			//chrono();  // Not yet implemented.
			//progress_timer();  // Deprecated.
			//cpu_timer();
		}

		{
			//factory();

			//smart_ptr();
			//any();
			//array_main();
			//tuple();
			//circular_buffer();
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
			//meta_state_machine();  // Not yet implemented.
			//state_machine_library();  // Not yet implemented.

			//thread();
			//process();
			//interprocess();  // Not yet implemented.
			iostreams();
		}

		// Mathematics & scientific computation.
		{
			//units();

			//math_constants();
			//math_floating_point_utilities();
			//math_special_functions();
			//math_statistical_distributions();
			//accumulator();
			//random();
			//ublas();
			//odeint();
			//math_bindings();
			//numeric_bindings();

			//multiprecision();  // Not yet implemented.

			//polygon();
			//geometry();

			//graph();
			//graph_parallel();  // Not yet implemented.
		}

		// Communication.
		{
			//asio_io_service();
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

		// High performance computing.
		{
			// Open Computing Language (OpenCL).
			//compute();

			// Message Passing Interface (MPI).
			//mpi();
		}

#if 0
		// Run a main loop.
		while (true)
		{
			if (std::cin.rdbuf() && std::cin.rdbuf()->in_avail() >= 0)
			{
				std::cout << "Press 'Q' to exit ..." << std::endl;

				const char ch = std::cin.rdbuf()->sgetc();
				if ('q' == ch || 'Q' == ch)
				{
					break;
				}

				char dummy;
				while (std::cin.readsome(&dummy, 1) > 0);
			}

			//std::this_thread::yield();
		}
#endif
	}
	catch (const std::bad_alloc &ex)
	{
		std::cerr << "std::bad_alloc caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &ex)
	{
		std::cerr << "std::exception caught: " << ex.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cerr << "Unknown exception caught." << std::endl;
		retval = EXIT_FAILURE;
	}

	std::cout << "Press any key to exit ..." << std::endl;
	std::cin.get();

	return retval;
}
