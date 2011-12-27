#include "stdafx.h"
#include "ConsoleWindow.h"


#ifdef _DEBUG
#define new DEBUG_NEW
#endif


ConsoleWindow::ConsoleWindow()
: isValid_(false)
{
}

ConsoleWindow::~ConsoleWindow()
{
}

/*static*/ ConsoleWindow & ConsoleWindow::getInstance()
{
	static ConsoleWindow console;
	return console;
}

/*static*/ void ConsoleWindow::initialize()
{
	if (AllocConsole())
	{
		freopen("CONOUT$", "w", stdout);
		freopen("CONIN$", "r", stdin);

		//
#if defined(_UNICODE) || defined(UNICODE)
		SetConsoleTitle(L"console window");
#else
		SetConsoleTitle("console window");
#endif

		ConsoleWindow::getInstance().isValid_ = true;
	}
}

/*static*/ void ConsoleWindow::finalize()
{
	if (ConsoleWindow::getInstance().isValid_)
		FreeConsole();

	ConsoleWindow::getInstance().isValid_ = false;
}
