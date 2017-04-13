#include <memory>
#include <QApplication>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QPainterPath>
#include <QPainter>
#include <QMap>
#include <QDebug>
#include <QDir>
#include <QGuiApplication>
#include <QQmlEngine>
#include <QQmlApplicationEngine>
#include <QQmlComponent>
#include <QQmlProperty>
#include <QQmlFileSelector>
#include <QQuickView>  // Not using QQmlApplicationEngine because many examples don't have a Window{}.
#include <QQuickItem>
#include "MainWindow.h"
#include "SdiMainWindow.h"
#include "MdiMainWindow.h"
#include "OsgMainWindow.h"
#include "MySlotClass.h"


namespace {
namespace local {

void usage()
{
	qWarning() << "Usage: mainwindow [-SizeHint<color> <width>x<height>] ...";
	exit(1);
}

enum ParseCommandLineArgumentsResult {
	CommandLineArgumentsOk,
	CommandLineArgumentsError,
	HelpRequested
};

ParseCommandLineArgumentsResult parseCustomSizeHints(const QStringList &arguments, MainWindow::CustomSizeHintMap *result)
{
	result->clear();
	const int argumentCount = arguments.size();
	for (int i = 1; i < argumentCount; ++i)
	{
		const QString &arg = arguments.at(i);
		if (arg.startsWith(QLatin1String("-SizeHint"))) 
		{
			const QString name = arg.mid(9);
			if (name.isEmpty())
				return CommandLineArgumentsError;
			if (++i == argumentCount)
				return CommandLineArgumentsError;
			const QString sizeStr = arguments.at(i);
			const int idx = sizeStr.indexOf(QLatin1Char('x'));
			if (idx == -1)
				return CommandLineArgumentsError;
			bool ok;
			const int w = sizeStr.leftRef(idx).toInt(&ok);
			if (!ok)
				return CommandLineArgumentsError;
			const int h = sizeStr.midRef(idx + 1).toInt(&ok);
			if (!ok)
				return CommandLineArgumentsError;
			result->insert(name, QSize(w, h));
		}
		else if (arg == QLatin1String("-h") || arg == QLatin1String("--help"))
		{
			return HelpRequested;
		}
		else
		{
			return CommandLineArgumentsError;
		}
	}

	return CommandLineArgumentsOk;
}

// REF [file] >> ${QT5_HOME}/examples/widgets/mainwindows/mainwindow/main.cpp
// REF [site] >> http://doc.qt.io/qt-5/qtwidgets-mainwindows-mainwindow-example.html
int mainwindows_mainwindow_example(int argc, char* argv[])
{
	QApplication app(argc, argv);
	MainWindow::CustomSizeHintMap customSizeHints;
	switch (local::parseCustomSizeHints(QCoreApplication::arguments(), &customSizeHints))
	{
	case CommandLineArgumentsOk:
		break;
	case CommandLineArgumentsError:
		usage();
		return -1;
	case HelpRequested:
		usage();
		return 0;
	}

	MainWindow mainWin(customSizeHints);
	mainWin.resize(800, 600);
	mainWin.show();

	return app.exec();
}

// REF [file] >> ${QT5_HOME}/examples/widgets/mainwindows/sdi/main.cpp
// REF [site] >> http://doc.qt.io/qt-5/qtwidgets-mainwindows-sdi-example.html
int mainwindows_sdi_example(int argc, char* argv[])
{
	//Q_INIT_RESOURCE(sdi);  // Move to main().

	QApplication app(argc, argv);
	QCoreApplication::setApplicationName("SDI Example");
	QCoreApplication::setOrganizationName("QtProject");
	QCoreApplication::setApplicationVersion(QT_VERSION_STR);
	QCommandLineParser parser;
	parser.setApplicationDescription(QCoreApplication::applicationName());
	parser.addHelpOption();
	parser.addVersionOption();
	parser.addPositionalArgument("file", "The file(s) to open.");
	parser.process(app);

	SdiMainWindow *mainWin = Q_NULLPTR;
	foreach(const QString &file, parser.positionalArguments())
	{
		SdiMainWindow *newWin = new SdiMainWindow(file);
		newWin->tile(mainWin);
		newWin->show();
		mainWin = newWin;
	}

	if (!mainWin)
		mainWin = new SdiMainWindow;
	mainWin->show();

	return app.exec();
}

// REF [file] >> ${QT5_HOME}/examples/widgets/mainwindows/mdi/main.cpp
// REF [site] >> http://doc.qt.io/qt-5/qtwidgets-mainwindows-mdi-example.html
int mainwindows_mdi_example(int argc, char* argv[])
{
	//Q_INIT_RESOURCE(mdi);  // Move to main().

	QApplication app(argc, argv);
	QCoreApplication::setApplicationName("MDI Example");
	QCoreApplication::setOrganizationName("QtProject");
	QCoreApplication::setApplicationVersion(QT_VERSION_STR);
	QCommandLineParser parser;
	parser.setApplicationDescription("Qt MDI Example");
	parser.addHelpOption();
	parser.addVersionOption();
	parser.addPositionalArgument("file", "The file to open.");
	parser.process(app);

	MdiMainWindow mainWin;
	foreach(const QString &fileName, parser.positionalArguments())
		mainWin.openFile(fileName);
	mainWin.show();

	return app.exec();
}

int qtquick_simpler_example(int argc, char* argv[])
{
	QGuiApplication app(argc, argv);

	QQmlApplicationEngine engine;
	engine.load(QUrl(QStringLiteral("qrc:/SimplerQml.qml")));  // REF [file] >> ./SimplerQml.qrc.

	return app.exec();
}

int qtquick_simple_example(int argc, char* argv[])
{
	QGuiApplication app(argc, argv);

	QQmlApplicationEngine engine;
	engine.load(QUrl(QStringLiteral("qrc:/SimpleQml.qml")));  // REF [file] >> ./SimpleQml.qrc.

	return app.exec();
}

// REF [file] >> ${QT5_HOME}/examples/quick/demos/samegame/main.cpp
// REF [site] >> http://doc.qt.io/qt-5/qtquick-demos-samegame-main-cpp.html
int qtquick_samegame_example(int argc, char* argv[])
{
    QGuiApplication app(argc,argv);
    app.setOrganizationName("QtProject");
    app.setOrganizationDomain("qt-project.org");
    app.setApplicationName(QFileInfo(app.applicationFilePath()).baseName());

    QQuickView view;
    if (qgetenv("QT_QUICK_CORE_PROFILE").toInt())
	{
        QSurfaceFormat f = view.format();
        f.setProfile(QSurfaceFormat::CoreProfile);
        f.setVersion(4, 4);
        view.setFormat(f);
    }
    view.connect(view.engine(), &QQmlEngine::quit, &app, &QCoreApplication::quit);
	new QQmlFileSelector(view.engine(), &view);
	view.setSource(QUrl("qrc:///demos/samegame/samegame.qml"));  // REF [file] >> ./samegame.qrc.
	if (view.status() == QQuickView::Error)
        return -1;
    view.setResizeMode(QQuickView::SizeRootObjectToView);
    if (QGuiApplication::platformName() == QLatin1String("qnx") || QGuiApplication::platformName() == QLatin1String("eglfs"))
	{
        view.showFullScreen();
    }
	else
	{
        view.show();
    }

    return app.exec();
}

// REF [site] >> http://doc.qt.io/qt-5/qtquick-layouts-example.html
int qtquick_layouts_example(int argc, char* argv[])
{
	QGuiApplication app(argc, argv);

	QQmlApplicationEngine engine;
	//engine.load(QUrl(QStringLiteral("qrc:///layouts/layouts.qml")));
	engine.load(QUrl::fromLocalFile("data/gui/qt5/layouts.qml"));

	return app.exec();
}

// REF [site] >> http://doc.qt.io/qt-5/qtqml-cppintegration-interactqmlfromcpp.html
int qtquick_interaction_with_object_example(int argc, char* argv[])
{
	QGuiApplication app(argc, argv);

#if 0
	// Use QQmlComponent.
	QQmlEngine engine;
	//QQmlApplicationEngine engine;
	//QQmlComponent component(&engine, QUrl::fromLocalFile("data/gui/qt5/rectangle.qml"));
	QQmlComponent component(&engine, QUrl::fromLocalFile("data/gui/qt5/item.qml"));

	std::unique_ptr<QObject> root(component.create());

	// Show view.
	// Do something.
#else
	// Use QQuickView.
	QQuickView view;
#if 0
	// NOTICE [error] >> Runtime error.
	//	- QQuickView does not support using windows as a root item.
	//	- If you wish to create your root window from QML, consider using QQmlApplicationEngine instead.
	view.setSource(QUrl::fromLocalFile("data/gui/qt5/rectangle.qml"));
#else
	view.setSource(QUrl::fromLocalFile("data/gui/qt5/item.qml"));
#endif

	QObject *root = (QObject *)view.rootObject();
#endif

	// Access objects.
	root->setProperty("width", 300);
	QQmlProperty(root, "height").write(300);

	QQuickItem *item = qobject_cast<QQuickItem *>(root);
	if (item)
		item->setWidth(500);

	QObject *rect = root->findChild<QObject *>("rect");
	if (rect)
		rect->setProperty("color", "red");

	// Access properties.
	// Always use QObject::setProperty(), QQmlPerperty or QMetaProperty::write() to change a QML property vale, to ensure the QML engine is made aware of the property change.
	qDebug() << "Property value:" << QQmlProperty::read(root, "someNumber").toInt();
	QQmlProperty::write(root, "someNumber", 5000);

	qDebug() << "Property value:" << root->property("someNumber").toInt();
	root->setProperty("someNumber", 100);

	// Invoke QML methods.
	QVariant returnedValue;
	QVariant msg("Hello from C++");
	QMetaObject::invokeMethod(root, "myQmlFunction",
		Q_RETURN_ARG(QVariant, returnedValue),
		Q_ARG(QVariant, msg));

	qDebug() << "QML function returned:" << returnedValue.toString();

	// Connect to QML signals.
	MySlotClass myClass;
	QObject::connect(root, SIGNAL(qmlSignal(QString)), &myClass, SLOT(cppSlot(QString)));
	QObject::connect(root, SIGNAL(qmlSignalObject(QVariant)), &myClass, SLOT(cppSlotObject(QVariant)));

	view.show();

	return app.exec();
}

// REF [site] >> https://github.com/Submanifold/QtOSG/blob/master/QtOSG.cpp
int osg_integration_using_qtosg(int argc, char** argv)
{
	QApplication application(argc, argv);

	OsgMainWindow mainWindow;
	mainWindow.show();

	return application.exec();
}

}  // namespace local
}  // unnamed namespace

namespace my_qt5 {

void render_qt_text(QPainter *painter, int w, int h, const QColor &color)
{
	QPainterPath path;
	path.moveTo(-0.083695, 0.283849);
	path.cubicTo(-0.049581, 0.349613, -0.012720, 0.397969, 0.026886, 0.428917);
	path.cubicTo(0.066493, 0.459865, 0.111593, 0.477595, 0.162186, 0.482108);
	path.lineTo(0.162186, 0.500000);
	path.cubicTo(0.115929, 0.498066, 0.066565, 0.487669, 0.014094, 0.468810);
	path.cubicTo(-0.038378, 0.449952, -0.088103, 0.423839, -0.135082, 0.390474);
	path.cubicTo(-0.182061, 0.357108, -0.222608, 0.321567, -0.256722, 0.283849);
	path.cubicTo(-0.304712, 0.262250, -0.342874, 0.239362, -0.371206, 0.215184);
	path.cubicTo(-0.411969, 0.179078, -0.443625, 0.134671, -0.466175, 0.081963);
	path.cubicTo(-0.488725, 0.029255, -0.500000, -0.033043, -0.500000, -0.104932);
	path.cubicTo(-0.500000, -0.218407, -0.467042, -0.312621, -0.401127, -0.387573);
	path.cubicTo(-0.335212, -0.462524, -0.255421, -0.500000, -0.161752, -0.500000);
	path.cubicTo(-0.072998, -0.500000, 0.003903, -0.462444, 0.068951, -0.387331);
	path.cubicTo(0.133998, -0.312218, 0.166522, -0.217440, 0.166522, -0.102998);
	path.cubicTo(0.166522, -0.010155, 0.143394, 0.071325, 0.097138, 0.141441);
	path.cubicTo(0.050882, 0.211557, -0.009396, 0.259026, -0.083695, 0.283849);
	path.moveTo(-0.167823, -0.456963);
	path.cubicTo(-0.228823, -0.456963, -0.277826, -0.432624, -0.314831, -0.383946);
	path.cubicTo(-0.361665, -0.323340, -0.385082, -0.230335, -0.385082, -0.104932);
	path.cubicTo(-0.385082, 0.017569, -0.361376, 0.112025, -0.313964, 0.178433);
	path.cubicTo(-0.277248, 0.229368, -0.228534, 0.254836, -0.167823, 0.254836);
	path.cubicTo(-0.105088, 0.254836, -0.054496, 0.229368, -0.016045, 0.178433);
	path.cubicTo(0.029055, 0.117827, 0.051605, 0.028691, 0.051605, -0.088975);
	path.cubicTo(0.051605, -0.179562, 0.039318, -0.255803, 0.014744, -0.317698);
	path.cubicTo(-0.004337, -0.365409, -0.029705, -0.400548, -0.061362, -0.423114);
	path.cubicTo(-0.093018, -0.445680, -0.128505, -0.456963, -0.167823, -0.456963);
	path.moveTo(0.379011, -0.404739);
	path.lineTo(0.379011, -0.236460);
	path.lineTo(0.486123, -0.236460);
	path.lineTo(0.486123, -0.197292);
	path.lineTo(0.379011, -0.197292);
	path.lineTo(0.379011, 0.134913);
	path.cubicTo(0.379011, 0.168117, 0.383276, 0.190442, 0.391804, 0.201886);
	path.cubicTo(0.400332, 0.213330, 0.411246, 0.219052, 0.424545, 0.219052);
	path.cubicTo(0.435531, 0.219052, 0.446227, 0.215264, 0.456635, 0.207689);
	path.cubicTo(0.467042, 0.200113, 0.474993, 0.188910, 0.480486, 0.174081);
	path.lineTo(0.500000, 0.174081);
	path.cubicTo(0.488436, 0.210509, 0.471957, 0.237911, 0.450564, 0.256286);
	path.cubicTo(0.429170, 0.274662, 0.407054, 0.283849, 0.384215, 0.283849);
	path.cubicTo(0.368893, 0.283849, 0.353859, 0.279094, 0.339115, 0.269584);
	path.cubicTo(0.324371, 0.260074, 0.313530, 0.246534, 0.306592, 0.228965);
	path.cubicTo(0.299653, 0.211396, 0.296184, 0.184075, 0.296184, 0.147002);
	path.lineTo(0.296184, -0.197292);
	path.lineTo(0.223330, -0.197292);
	path.lineTo(0.223330, -0.215667);
	path.cubicTo(0.241833, -0.224049, 0.260697, -0.237992, 0.279922, -0.257495);
	path.cubicTo(0.299147, -0.276999, 0.316276, -0.300129, 0.331310, -0.326886);
	path.cubicTo(0.338826, -0.341070, 0.349523, -0.367021, 0.363400, -0.404739);
	path.lineTo(0.379011, -0.404739);
	path.moveTo(-0.535993, 0.275629);

	painter->translate(w / 2, h / 2);
	double scale = qMin(w, h) * 8 / 10.0;
	painter->scale(scale, scale);

	painter->setRenderHint(QPainter::Antialiasing);

	painter->save();
	painter->translate(.1, .1);
	painter->fillPath(path, QColor(0, 0, 0, 63));
	painter->restore();

	painter->setBrush(color);
	painter->setPen(QPen(Qt::black, 0.02, Qt::SolidLine, Qt::FlatCap, Qt::RoundJoin));
	painter->drawPath(path);
}

int osgqt_widgets_example(int argc, char **argv);
int osgqt_vierwer_example(int argc, char **argv);

}  // namespace my_qt5

// Meta Object Compiler (moc).
//  moc object_name.h -o moc_object_name.cpp
// Resource Compiler (rcc).
//	(Optional) rcc -binary resource_name.qrc -o resource_name.rcc
//	rcc resource_name.qrc -name resource_name -o qrc_resource_name.cpp
// Build Qt Project.
//	qmake project_name.pro -spec win32-msvc2015
//	make or jom

int qt5_main(int argc, char* argv[])
{
	// ---------------------------------------------------------------
	//const int retval = local::mainwindows_mainwindow_example(argc, argv);

	// SDI -----------------------------------------------------------
	//Q_INIT_RESOURCE(sdi);  // Move from local::mainwindows_sdi_example() to here.
	//const int retval = local::mainwindows_sdi_example(argc, argv);

	// MDI -----------------------------------------------------------
	//Q_INIT_RESOURCE(mdi);  // Move from local::mainwindows_mdi_example() to here.
	//const int retval = local::mainwindows_mdi_example(argc, argv);

	// Qt Quick ------------------------------------------------------
	//const int retval = local::qtquick_simpler_example(argc, argv);
	//const int retval = local::qtquick_simple_example(argc, argv);

	const int retval = local::qtquick_layouts_example(argc, argv);
	//const int retval = local::qtquick_samegame_example(argc, argv);

	//const int retval = local::qtquick_interaction_with_object_example(argc, argv);

	// Integration with OpenSceneGraph -------------------------------
	//const int retval = my_qt5::osgqt_widgets_example(argc, argv);  // Compile-time error. I guess this example is for mobile devices.
	//const int retval = my_qt5::osgqt_vierwer_example(argc, argv);
	
	//const int retval = local::osg_integration_using_qtosg(argc, argv);

	return retval;
}
