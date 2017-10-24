#include <QtGui>
#include <QApplication>
#include "MyWidget.h"


// Usage.
//	- Generate a project.
//		qmake -project
//			The generated project filename is its directory name.
//	- (Optional) edit the project file.
//		e.g.)
//			QT += widgets gui
//			CONFIG += c++11
//	- Generate Makefile.
//		qmake
//		qmake project_name.pro
//		qmake -spec win32-msvc2015 project_name.pro
//		qmake -spec win32-msvc2015 -tp vc project_name.pro
//	- Make.
//		jom
//		make
//		nmake
//			Generate ui, moc, qrc files.

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);

	MyWidget widget;
	widget.show();

	return app.exec();
}
