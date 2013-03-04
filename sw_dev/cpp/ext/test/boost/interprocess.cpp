#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#if defined(__linux) || defined(__linux__) || defined(linux) || defined(__unix) || defined(__unix__) || defined(unix)
#elif defined(_WINDOWS) || defined(WIN32)
#include <boost/interprocess/managed_windows_shared_memory.hpp>
#endif
#include <iostream>
#include <stdexcept>


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

void interprocess()
{
	throw std::runtime_error("not yet implemented");
}
