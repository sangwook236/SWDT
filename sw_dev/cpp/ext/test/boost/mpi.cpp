#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#include <iostream>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

void mpi()
{
	boost::mpi::environment env;
	boost::mpi::communicator world;
	std::cout << "I am process " << world.rank() << " of " << world.size() << "." << std::endl;
}
