TEMPLATE = app

QT += qml quick gui
CONFIG += c++11

Release:DESTDIR = ../../../bin/x64/release
Release:OBJECTS_DIR = x64/release
Release:MOC_DIR = x64/release
Release:RCC_DIR = x64/release
Release:UI_DIR = x64/release

Debug:DESTDIR = ../../../bin/x64/debug
Debug:OBJECTS_DIR = x64/debug
Debug:MOC_DIR = x64/debug
Debug:RCC_DIR = x64/debug
Debug:UI_DIR = x64/debug

SOURCES += qt5_main.cpp \
    ColorSwatch.cpp \
    MainWindow.cpp \
    MdiChild.cpp \
    MdiMainWindow.cpp \
    OsgMainWindow.cpp \
    OsgPickHandler.cpp \
    osgQtWidgets.cpp \
    osgviewerQt.cpp \
    OSGWidget.cpp \
    SdiMainWindow.cpp \
    Toolbar.cpp

HEADERS += \
    ColorSwatch.h \
    MainWindow.h \
    MdiChild.h \
    MdiMainWindow.h \
    MySlotClass.h \
    OsgMainWindow.h \
    OsgPickHandler.h \
    OSGWidget.h \
    SdiMainWindow.h \
    Toolbar.h

RESOURCES += \
	MainWindow.qrc \
	mdi.qrc \
	samegame.qrc \
	#sdu.qrc \
	SimpleQml.qrc \
	SimplerQml.qrc

# Additional import path used to resolve QML modules in Qt Creator's code model
QML_IMPORT_PATH =

# Additional import path used to resolve QML modules just for Qt Quick Designer
QML_DESIGNER_IMPORT_PATH =

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

INCLUDEPATH += ../../../inc
LIBS += -L../../../lib

win32 {
INCLUDEPATH += D:/usr/local64/include
LIBS += -LD:/usr/local64/lib
#CONFIG(release, debug|release): LIBS += -lopencv_imgproc310 -lopencv_imgcodecs310 -lopencv_highgui310 -lopencv_core310
#else:CONFIG(debug, debug|release): LIBS += -lopencv_imgproc310d -lopencv_imgcodecs310d -lopencv_highgui310d -lopencv_core310d
}
unix {
INCLUDEPATH += /usr/local/include
LIBS += -L/usr/local/lib
#LIBS += -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_core
}

DISTFILES +=

#message(QTDIR = $$(QTDIR))
#MY_USR = $$(HOME)/my_usr
#message(MY_USR = $${MY_USR})
