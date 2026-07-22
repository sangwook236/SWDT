#include <stdexcept>
#include <iostream>
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <windows.h>
#include <dbghelp.h>
#elif defined(__linux__) || defined(__linux)
#include <csignal>
#include <sys/resource.h>
#endif

#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#pragma comment(lib, "dbghelp.lib")
#endif


namespace {
namespace local {

#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
void write_minidump(EXCEPTION_POINTERS* ep)
{
	HANDLE hFile = CreateFileA(
		"crash.dmp",
		GENERIC_WRITE, 0, nullptr,
		CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr
	);
	if (hFile == INVALID_HANDLE_VALUE) return;

	MINIDUMP_EXCEPTION_INFORMATION mei{};
	mei.ThreadId = GetCurrentThreadId();
	mei.ExceptionPointers = ep;
	mei.ClientPointers = FALSE;

	MiniDumpWriteDump(
		GetCurrentProcess(), GetCurrentProcessId(),
		hFile,
		MiniDumpNormal,
		//MiniDumpWithDataSegs | MiniDumpWithThreadInfo,
		//MiniDumpWithFullMemory,
		&mei, nullptr, nullptr
	);
	CloseHandle(hFile);
	std::cout << "Minidump written to crash.dmp" << std::endl;
}

LONG WINAPI unhandled_exception_filter(EXCEPTION_POINTERS* ep)
{
	write_minidump(ep);
	return EXCEPTION_EXECUTE_HANDLER;
}
#elif defined(__linux__) || defined(__linux)
void enable_core_dump()
{
    struct rlimit rl{};
    rl.rlim_cur = RLIM_INFINITY;
    rl.rlim_max = RLIM_INFINITY;
    setrlimit(RLIMIT_CORE, &rl);
}

void signal_handler(int sig)
{
    std::cerr << "Signal " << sig << " received, generating core dump..." << std::endl;
    signal(sig, SIG_DFL);  // Restore default handler and re-raise
    raise(sig);
}
#endif

void cause_access_violation()
{
	std::cout << "Causing access violation..." << std::endl;
	int* ptr = nullptr;
	*ptr = 5;
}

void cause_division_by_zero()
{
	std::cout << "Causing division by zero..." << std::endl;
	volatile int zero = 0;
	[[maybe_unused]] int result = 1 / zero;
}

void cause_stack_overflow()
{
	std::cout << "Causing stack overflow..." << std::endl;
	cause_stack_overflow();
}

#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
void cause_uncaught_exception()
{
	std::cout << "Causing uncaught exception..." << std::endl;
	throw std::runtime_error("Uncaught exception.");
}
#elif defined(__linux__) || defined(__linux)
void cause_abort()
{
	std::cout << "Causing abort..." << std::endl;
	abort();
}
#endif

}  // namespace local
}  // unnamed namespace

void core_dump()
{
	// Usage (Windows):
	//	1. Visual Studio
	//		Drag and drop the generated crash.dmp file into Visual Studio to analyze.
	//		Click "Debug with Native Only" to start debugging the dump.
	//		.pdb files are required in the same directory as the executable for proper symbol resolution
	//	2. WinDbg
	//		Install WinDbg or WinDbg Preview.
	//			winget install Microsoft.WinDbg
	//		Open the generated crash.dmp file in WinDbg.
	//			windbg -z crash.dmp
	//				Commands:
	//					!analyze -v          # Automatically analyze crash cause (run this first)
	//					k                    # Call stack of the current thread
	//					~*k                  # Call stack of all threads
	//					.ecxr                # Switch to the context at the point of exception
	//					lm                   # List of loaded modules
	//					dt <type> <addr>     # Display data type at a specific address

	// Usage (Linux):
	//	1. Enable core dump.
	//		Linux disables core dumps by default.
	//
	//		Check current core dump size limit (0 means disabled):
	//			ulimit -c
	//		Enable unlimited core dumps (current session):
	//			ulimit -c unlimited
	//	2. Build with debug symbols.
	//		g++ -g -o crash_test core_dump.cpp
	//	3. Analyze with GDB.
	//		Open core dump with GDB:
	//			gdb ./crash_test /tmp/core-crash_test-1234-...

#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
	SetUnhandledExceptionFilter(local::unhandled_exception_filter);
#elif defined(__linux__) || defined(__linux)
	enable_core_dump();

	// Register signal handlers
	signal(SIGSEGV, signal_handler);  // Access violation
	signal(SIGFPE,  signal_handler);  // Division by zero
	signal(SIGABRT, signal_handler);  // abort()
	signal(SIGILL,  signal_handler);  // Illegal instruction
#endif

	std::cout << "Select a crash type:\n"
		<< "1: Access Violation\n"
		<< "2: Division by zero\n"
		<< "3: Stack Overflow\n"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
		<< "4: Uncaught Exception\n"
#elif defined(__linux__) || defined(__linux)
		<< "4: Abort\n"
#endif
		<< "Enter your choice: ";

	int choice;
	std::cin >> choice;

	switch (choice)
	{
	case 1: local::cause_access_violation(); break;
	case 2: local::cause_division_by_zero(); break;
	case 3: local::cause_stack_overflow(); break;
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
	case 4: local::cause_uncaught_exception(); break;
#elif defined(__linux__) || defined(__linux)
	case 4: local::cause_abort(); break;
#endif
	default: std::cout << "Invalid choice.\n"; break;
	}
}
