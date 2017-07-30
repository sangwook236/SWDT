#include <memory>
#include <QGuiApplication>
#include <Qt3DCore/QEntity>
#include <Qt3DCore/QTransform>
#include <Qt3DRender/QCamera>
#include <Qt3DRender/QCameraLens>
#include <Qt3DExtras/Qt3DWindow.h>
#include <Qt3DExtras/QPhongMaterial>
#include <Qt3DExtras/QDiffuseSpecularMapMaterial>
#include <Qt3DExtras/QCylinderMesh>
#include <Qt3DExtras/QSphereMesh>
#include <Qt3DExtras/QTorusMesh>
#include <Qt3DExtras/QFirstPersonCameraController.h>
#include <QPropertyAnimation>


namespace {
namespace local {

// REF [file] >> ${QT5_HOME}/Examples/Qt-5.8/qt3d/simple-cpp/main.cpp
// REF [file] >> ${QT5_HOME}/Examples/Qt-5.8/qt3d/basicshapes-cpp
Qt3DCore::QEntity *createScene(Qt3DExtras::Qt3DWindow *view)
{
	// Root entity.
	Qt3DCore::QEntity *rootEntity = new Qt3DCore::QEntity;

	// Camera.
	Qt3DRender::QCamera *cameraEntity = view->camera();
	cameraEntity->lens()->setPerspectiveProjection(45.0f, 16.0f / 9.0f, 0.1f, 1000.0f);
	cameraEntity->setPosition(QVector3D(0, 0, -60.0f));
	cameraEntity->setUpVector(QVector3D(0, 1, 0));
	cameraEntity->setViewCenter(QVector3D(0, 0, 0));

	// For camera controls.
	Qt3DExtras::QFirstPersonCameraController *camController = new Qt3DExtras::QFirstPersonCameraController(rootEntity);
	camController->setCamera(cameraEntity);

	// Material.
	Qt3DExtras::QPhongMaterial *material1 = new Qt3DExtras::QPhongMaterial(rootEntity);
	material1->setAmbient(Qt::red);
	material1->setDiffuse(Qt::black);
	Qt3DExtras::QPhongMaterial *material2 = new Qt3DExtras::QPhongMaterial(rootEntity);
	material2->setAmbient(Qt::blue);
	material2->setDiffuse(Qt::black);

	// Torus.
	Qt3DCore::QEntity *torusEntity = new Qt3DCore::QEntity(rootEntity);
	Qt3DExtras::QTorusMesh *torusMesh = new Qt3DExtras::QTorusMesh;
	torusMesh->setRadius(5);
	torusMesh->setMinorRadius(1);
	torusMesh->setRings(100);
	torusMesh->setSlices(20);

	Qt3DCore::QTransform *torusTransform = new Qt3DCore::QTransform;
	torusTransform->setScale3D(QVector3D(1.5, 1, 0.5));
	torusTransform->setRotation(QQuaternion::fromAxisAndAngle(QVector3D(1, 0, 0), 45.0f));

	torusEntity->addComponent(torusMesh);
	torusEntity->addComponent(torusTransform);
	torusEntity->addComponent(material1);

	// Sphere.
	Qt3DCore::QEntity *sphereEntity = new Qt3DCore::QEntity(rootEntity);
	Qt3DExtras::QSphereMesh *sphereMesh = new Qt3DExtras::QSphereMesh;
	sphereMesh->setRadius(3);

	Qt3DCore::QTransform *sphereTransform = new Qt3DCore::QTransform;
	sphereTransform->setScale3D(QVector3D(0.5, 1, 1.5));
	sphereTransform->setRotation(QQuaternion::fromAxisAndAngle(QVector3D(1, 0, 0), 45.0f));

	sphereEntity->addComponent(sphereMesh);
	sphereEntity->addComponent(sphereTransform);
	sphereEntity->addComponent(material2);

	return rootEntity;
}

}  // namespace local
}  // unnamed namespace

namespace my_qt5 {

int qt3d(int argc, char **argv)
{
	QGuiApplication app(argc, argv);
	Qt3DExtras::Qt3DWindow view;

	Qt3DCore::QEntity *rootEntity = local::createScene(&view);

	view.setRootEntity(rootEntity);
	view.show();

	return app.exec();
}

}  // namespace my_qt5
