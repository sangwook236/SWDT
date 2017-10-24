#include <QtGui>
#include <QLabel>
#include <QApplication>


// Usage.
//	- Generate a project.
//		qmake -project
//			The generated project filename is its directory name.
//	- Generate Makefile.
//		qmake project_name.pro
//		qmake -spec win32-msvc2015 project_name.pro
//		qmake -spec win32-msvc2015 -tp vc project_name.pro
//	- Make.
//		jom
//		make
//		nmake

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);

	QLabel *label = new QLabel("Hello World !!!");
	label->show();

	return app.exec();
}
