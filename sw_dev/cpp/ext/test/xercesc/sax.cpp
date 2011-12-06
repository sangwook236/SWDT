#include "stdafx.h"
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>


#if defined(XERCES_NEW_IOSTREAMS)
#include <iostream>
#else
#include <iostream.h>
#endif

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

XERCES_CPP_NAMESPACE_USE

int sax()
{
	return 0;
}
