#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <iostream>


int main(int argc, char* argv[])
{
	int lapack_main(int argc, char* argv[]);
	int atlas_main(int argc, char* argv[]);
	int eigen_main(int argc, char* argv[]);
	int armadillo_main(int argc, char* argv[]);
	int cvm_main(int argc, char* argv[]);
	int mtl_main(int argc, char* argv[]);

	try
	{
		lapack_main(argc, argv);
		
		atlas_main(argc, argv);
		eigen_main(argc, argv);
		armadillo_main(argc, argv);
		
		cvm_main(argc, argv);

		//mtl_main(argc, argv);  // not yet implemented
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

