#include "MyWindow.h"
#include <QApplication>
#include <QFont>
#include <QPushButton>
#include <iostream>
#include <stdexcept>
#include <cstdlib>


namespace {
namespace local {

int very_simple_example(int argc, char* argv[])
{
    QApplication app(argc, argv);

    QWidget window;
    window.setFixedSize(400, 300);
    window.setWindowTitle("Qt4 Application");

    QPushButton btnQuit("Quit", &window);
    btnQuit.setFont(QFont("Times", 18, QFont::Bold));
    btnQuit.setGeometry(10, 10, 80, 30);
    btnQuit.resize(160, 30);

    QObject::connect(&btnQuit, SIGNAL(clicked()), &app, SLOT(quit()));

    window.show();

    return app.exec();
}

// Meta Object Compiler (moc).
//  moc-qt4 MyWindow.h -o moc_MyWindows.cpp
int simple_example(int argc, char* argv[])
{
    QApplication app(argc, argv);

    MyWindow window;
    QObject::connect(&window, SIGNAL(areaChanged()), &window, SLOT(handAreaChanged()));
    QObject::connect(&window, SIGNAL(quitButtonClicked()), &app, SLOT(quit()));

    window.show();

    window.setArea(100);

    return app.exec();
}

}
}

int main(int argc, char* argv[])
{
	int retval = EXIT_SUCCESS;
	try
	{
	    //retval = local::very_simple_example(argc, argv);
	    retval = local::simple_example(argc, argv);
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

	//std::cout << "press any key to exit ..." << std::endl;
	//std::cin.get();

	return retval;
}
