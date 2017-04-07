//--S [] 2017/04/06: Sang-Wook Lee.
#if defined(_WIN64) || defined(WIN64)
#include <windows.h>
#endif
//--E [] 2017/04/06: Sang-Wook Lee.
#include <memory>
#include <QTimer>
#include <QApplication>
#include <QGridLayout>
#include <QQmlEngine>
#include <QQmlApplicationEngine>
#include <QQmlComponent>
#include <QQmlProperty>
#include <QQuickWidget>
#include <QQuickItem>
#include <QDebug>
#include <osgViewer/CompositeViewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgGA/MultiTouchTrackballManipulator>
#include <osgDB/ReadFile>
#include <osgQt/GraphicsWindowQt>
#include <iostream>

namespace {
namespace local {

class OsgViewerWidget : public QWidget, public osgViewer::CompositeViewer
{
public:
	OsgViewerWidget(QWidget *parent = 0, Qt::WindowFlags f = 0, osgViewer::ViewerBase::ThreadingModel threadingModel = osgViewer::CompositeViewer::SingleThreaded)
	: QWidget(parent, f)
    {
        setThreadingModel(threadingModel);

        // Disable the default setting of viewer.done() by pressing Escape.
        setKeyEventSetsDone(0);

        QWidget *widget1 = addViewWidget(createGraphicsWindow(0, 0, 100, 100), osgDB::readRefNodeFile("./data/graphics_3d/osg/cow.osgt"));
        QWidget *widget2 = addViewWidget(createGraphicsWindow(0, 0, 100, 100), osgDB::readRefNodeFile("./data/graphics_3d/osg/glider.osgt"));
        QWidget *widget3 = addViewWidget(createGraphicsWindow(0, 0, 100, 100), osgDB::readRefNodeFile("./data/graphics_3d/osg/axes.osgt"));
        QWidget *widget4 = addViewWidget(createGraphicsWindow(0, 0, 100, 100), osgDB::readRefNodeFile("./data/graphics_3d/osg/fountain.osgt"));
        QWidget *popupWidget = addViewWidget(createGraphicsWindow(900, 100, 320, 240, "Popup window", true), osgDB::readRefNodeFile("./data/graphics_3d/osg/dumptruck.osgt"));
        popupWidget->show();

        QGridLayout *grid = new QGridLayout;
        grid->addWidget(widget1, 0, 0);
        grid->addWidget(widget2, 0, 1);
        grid->addWidget(widget3, 1, 0);
        grid->addWidget(widget4, 1, 1);
        setLayout(grid);

        connect(&timer_, SIGNAL(timeout()), this, SLOT(update()));
        timer_.start(10);
    }

    QWidget * addViewWidget(osgQt::GraphicsWindowQt* gw, osg::ref_ptr<osg::Node> scene)
    {
        osgViewer::View *view = new osgViewer::View;
        addView(view);

        osg::Camera *camera = view->getCamera();
        camera->setGraphicsContext(gw);

        const osg::GraphicsContext::Traits *traits = gw->getTraits();

        camera->setClearColor(osg::Vec4(0.2, 0.2, 0.6, 1.0));
        camera->setViewport(new osg::Viewport(0, 0, traits->width, traits->height));
        camera->setProjectionMatrixAsPerspective(30.0f, static_cast<double>(traits->width) / static_cast<double>(traits->height), 1.0f, 10000.0f);

        view->setSceneData(scene);
        view->addEventHandler(new osgViewer::StatsHandler);
        view->setCameraManipulator(new osgGA::MultiTouchTrackballManipulator);
        gw->setTouchEventsEnabled(true);

        return gw->getGLWidget();
    }

    osgQt::GraphicsWindowQt * createGraphicsWindow(int x, int y, int w, int h, const std::string &name = "", bool windowDecoration = false)
    {
        osg::DisplaySettings *ds = osg::DisplaySettings::instance().get();
        osg::ref_ptr<osg::GraphicsContext::Traits> traits = new osg::GraphicsContext::Traits;
        traits->windowName = name;
        traits->windowDecoration = windowDecoration;
        traits->x = x;
        traits->y = y;
        traits->width = w;
        traits->height = h;
        traits->doubleBuffer = true;
        traits->alpha = ds->getMinimumNumAlphaBits();
        traits->stencil = ds->getMinimumNumStencilBits();
        traits->sampleBuffers = ds->getMultiSamples();
        traits->samples = ds->getNumMultiSamples();

        return new osgQt::GraphicsWindowQt(traits.get());
    }

    virtual void paintEvent(QPaintEvent * /*event*/)
    { frame(); }

protected:
    QTimer timer_;
};

}  // namespace local
}  // unnamed namespace

namespace my_qt5 {
	
// REF [file] >> ${osgQt_HOME}/examples/osgviewerQt/osgviewerQt.cpp
int osgqt_vierwer_example(int argc, char **argv)
{
	osg::ArgumentParser arguments(&argc, argv);

#if QT_VERSION >= 0x050000
	// Qt5 is currently crashing and reporting "Cannot make QOpenGLContext current in a different thread" when the viewer is run multi-threaded, this is regression from Qt4
	osgViewer::ViewerBase::ThreadingModel threadingModel = osgViewer::ViewerBase::SingleThreaded;
#else
	osgViewer::ViewerBase::ThreadingModel threadingModel = osgViewer::ViewerBase::CullDrawThreadPerContext;
#endif

	while (arguments.read("--SingleThreaded")) threadingModel = osgViewer::ViewerBase::SingleThreaded;
	while (arguments.read("--CullDrawThreadPerContext")) threadingModel = osgViewer::ViewerBase::CullDrawThreadPerContext;
	while (arguments.read("--DrawThreadPerContext")) threadingModel = osgViewer::ViewerBase::DrawThreadPerContext;
	while (arguments.read("--CullThreadPerCameraDrawThreadPerContext")) threadingModel = osgViewer::ViewerBase::CullThreadPerCameraDrawThreadPerContext;

#if QT_VERSION >= 0x040800
	// Required for multithreaded QGLWidget on Linux/X11, see http://blog.qt.io/blog/2011/06/03/threaded-opengl-in-4-8/
	if (threadingModel != osgViewer::ViewerBase::SingleThreaded)
		QApplication::setAttribute(Qt::AA_X11InitThreads);
#endif

#if 0
	QApplication app(argc, argv);

	std::unique_ptr<local::OsgViewerWidget> osgViewerWidget(new local::OsgViewerWidget(0, Qt::Widget, threadingModel));
	osgViewerWidget->setGeometry(100, 100, 800, 600);
	osgViewerWidget->show();
#elif 1
	QApplication app(argc, argv);

	// Use QQuickWidget.
	QQuickWidget view(QUrl::fromLocalFile("data/gui/qt5/rectangle.qml"));

	local::OsgViewerWidget *osgViewerWidget = new local::OsgViewerWidget(0, Qt::Widget, threadingModel);
	//osgViewerWidget->setGeometry(100, 100, 800, 600);
	//osgViewerWidget->show();

	QGridLayout *grid = new QGridLayout;
	grid->addWidget(osgViewerWidget, 0, 0);
	view.setLayout(grid);

	view.show();
#else
	QApplication app(argc, argv);

	// Use QQuickWidget.
	QQuickWidget view(QUrl::fromLocalFile("data/gui/qt5/rectangle.qml"));

	QQuickItem *root = view.rootObject();
	if (!root)
	{
		qDebug() << "Root object not found !!!";
		return -1;
	}

	//QObject *region = root->findChild<QObject *>("rectangle1");
	QQuickItem *region = root->findChild<QQuickItem *>("rectangle1");
	//QQuickWidget *region = root->findChild<QQuickWidget *>("rectangle1");  // Runtime error: casting error.
	if (!region)
	{
		qDebug() << "Object not found !!!";
		return -1;
	}

	local::OsgViewerWidget *osgViewerWidget = new local::OsgViewerWidget(0, Qt::Widget, threadingModel);
	//osgViewerWidget->setGeometry(100, 100, 800, 600);
	//osgViewerWidget->show();

	// FIXME [fix] >> Not working.

	QGridLayout *grid = new QGridLayout;
	grid->addWidget(osgViewerWidget, 0, 0);

	QQuickWidget widget(&view);
	widget.setLayout(grid);
	widget.show();
	QQuickItem *abc = widget.rootObject();
	abc->setParent(region);

	view.show();
#endif

	return app.exec();
}

}  // namespace my_qt5
