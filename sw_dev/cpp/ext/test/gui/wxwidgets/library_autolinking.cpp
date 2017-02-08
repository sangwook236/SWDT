#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)

#	if defined(_MSC_VER)

#		if defined(DEBUG) || defined(_DEBUG)

#pragma comment(lib, "wxmsw31ud_adv.lib")
#pragma comment(lib, "wxmsw31ud_aui.lib")
#pragma comment(lib, "wxmsw31ud_gl.lib")
#pragma comment(lib, "wxmsw31ud_html.lib")
#pragma comment(lib, "wxmsw31ud_media.lib")
#pragma comment(lib, "wxmsw31ud_propgrid.lib")
#pragma comment(lib, "wxmsw31ud_qa.lib")
#pragma comment(lib, "wxmsw31ud_ribbon.lib")
#pragma comment(lib, "wxmsw31ud_richtext.lib")
#pragma comment(lib, "wxmsw31ud_stc.lib")
#pragma comment(lib, "wxmsw31ud_webview.lib")
#pragma comment(lib, "wxmsw31ud_xrc.lib")
#pragma comment(lib, "wxmsw31ud_core.lib")
#pragma comment(lib, "wxbase31ud_net.lib")
#pragma comment(lib, "wxbase31ud_xml.lib")
#pragma comment(lib, "wxbase31ud.lib")

#		else

#pragma comment(lib, "wxmsw31u_adv.lib")
#pragma comment(lib, "wxmsw31u_aui.lib")
#pragma comment(lib, "wxmsw31u_gl.lib")
#pragma comment(lib, "wxmsw31u_html.lib")
#pragma comment(lib, "wxmsw31u_media.lib")
#pragma comment(lib, "wxmsw31u_propgrid.lib")
#pragma comment(lib, "wxmsw31u_qa.lib")
#pragma comment(lib, "wxmsw31u_ribbon.lib")
#pragma comment(lib, "wxmsw31u_richtext.lib")
#pragma comment(lib, "wxmsw31u_stc.lib")
#pragma comment(lib, "wxmsw31u_webview.lib")
#pragma comment(lib, "wxmsw31u_xrc.lib")
#pragma comment(lib, "wxmsw31u_core.lib")
#pragma comment(lib, "wxbase31u_net.lib")
#pragma comment(lib, "wxbase31u_xml.lib")
#pragma comment(lib, "wxbase31u.lib")

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
