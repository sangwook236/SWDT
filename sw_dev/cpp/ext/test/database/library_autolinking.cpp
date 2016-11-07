#if defined(_WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "libsoci_odbc_4_0d.lib")
#pragma comment(lib, "libsoci_sqlite3_4_0d.lib")
#pragma comment(lib, "libsoci_empty_4_0d.lib")
#pragma comment(lib, "libsoci_core_4_0d.lib")
//#pragma comment(lib, "liblmdb_d.lib")
//#pragma comment(lib, "sqlited.lib")

#		else

#pragma comment(lib, "libsoci_odbc_4_0.lib")
#pragma comment(lib, "libsoci_sqlite3_4_0.lib")
#pragma comment(lib, "libsoci_empty_4_0.lib")
#pragma comment(lib, "libsoci_core_4_0.lib")
//#pragma comment(lib, "liblmdb.lib")
//#pragma comment(lib, "sqlite.lib")

#		endif

#	else

//#error [SWDT] not supported compiler

#	endif

#elif defined(__MINGW32__)

#	if defined(__GUNC__)

#		if defined(DEBUG) || defined(_DEBUG)
#		else
#		endif

#	else

//#error [SWDT] not supported compiler

#	endif

#elif defined(__CYGWIN__)

#	if defined(__GUNC__)

#		if defined(DEBUG) || defined(_DEBUG)
#		else
#		endif

#	else

//#error [SWDT] not supported compiler

#	endif

#elif defined(__linux) || defined(__linux__) || defined(linux) || defined(__unix) || defined(__unix__) || defined(unix)

#	if defined(__GUNC__)

#		if defined(DEBUG) || defined(_DEBUG)
#		else
#		endif

#	else

//#error [SWDT] not supported compiler

#	endif

#elif defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__ ) || defined(__DragonFly__)

#	if defined(__GUNC__)

#		if defined(DEBUG) || defined(_DEBUG)
#		else
#		endif

#	else

//#error [SWDT] not supported compiler

#	endif

#elif defined(__APPLE__)

#	if defined(__GUNC__)

#		if defined(DEBUG) || defined(_DEBUG)
#		else
#		endif

#	else

//#error [SWDT] not supported compiler

#	endif

#else

#error [SWDT] not supported operating sytem

#endif
