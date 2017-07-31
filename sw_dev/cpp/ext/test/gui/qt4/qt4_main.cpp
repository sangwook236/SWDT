#include "MyWindow.h"
#include "MyMainWindow.h"
#include <QApplication>
#include <QFont>
#include <QPushButton>


namespace {
namespace local {

int very_simple_example(int argc, char* argv[])
{
    QApplication app(argc, argv);

    QWidget window;
    window.setFixedSize(400, 300);
    window.setWindowTitle("Qt4 Application #1");

    QPushButton btnQuit("Quit", &window);
    btnQuit.setFont(QFont("Times", 18, QFont::Bold));
    btnQuit.setGeometry(10, 10, 80, 30);
    btnQuit.resize(160, 30);

    QObject::connect(&btnQuit, SIGNAL(clicked()), &app, SLOT(quit()));

    window.show();

    return app.exec();
}

int simple_example_1(int argc, char* argv[])
{
    QApplication app(argc, argv);

    MyWindow window;
    //QObject::connect(&window, SIGNAL(areaChanged()), &window, SLOT(handAreaChanged()));
    QObject::connect(&window, SIGNAL(applicationQuitSignaled()), &app, SLOT(quit()));

    window.show();

    window.setArea(100);

    return app.exec();
}

int simple_example_2(int argc, char* argv[])
{
    QApplication app(argc, argv);

    MyMainWindow window;
    QObject::connect(&window, SIGNAL(applicationQuitSignaled()), &app, SLOT(quit()));

    window.show();

    return app.exec();
}

}  // namespace local
}  // unnamed namespace

// Usage.
//	- Compile meta objects by Meta Object Compiler (moc).
//		moc object_name.h -o moc_object_name.cpp
//	- Compile resources by Resource Compiler (rcc).
//		(optional) rcc -binary resource_name.qrc -o resource_name.rcc
//		rcc resource_name.qrc -name resource_name -o qrc_resource_name.cpp
//	- Generate Makefile.
//		qmake -spec win32-msvc2015 project_name.pro
//		qmake -spec win32-msvc2015 -tp vc project_name.pro
//	- Make.
//		jom
//		make
//		nmake

int qt4_main(int argc, char* argv[])
{
    //const int retval = local::very_simple_example(argc, argv);
    //const int retval = local::simple_example_1(argc, argv);
    const int retval = local::simple_example_2(argc, argv);
    
    return retval;
}
