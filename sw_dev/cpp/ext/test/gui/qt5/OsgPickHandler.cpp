//--S [] 2017/04/05: Sang-Wook Lee.
#if defined(_WIN64) || defined(WIN64)
#include <windows.h>
#endif
//--E [] 2017/04/05: Sang-Wook Lee.
#include "OsgPickHandler.h"

#include <osg/io_utils>

#include <osgUtil/IntersectionVisitor>
#include <osgUtil/LineSegmentIntersector>

#include <osgViewer/Viewer>

#include <iostream>

OsgPickHandler::~OsgPickHandler()
{
}

bool OsgPickHandler::handle( const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter& aa )
{
  if( ea.getEventType() != osgGA::GUIEventAdapter::RELEASE &&
      ea.getButton()    != osgGA::GUIEventAdapter::LEFT_MOUSE_BUTTON )
  {
    return false;
  }

  osgViewer::View* viewer = dynamic_cast<osgViewer::View*>( &aa );

  if( viewer )
  {
    osgUtil::LineSegmentIntersector* intersector
        = new osgUtil::LineSegmentIntersector( osgUtil::Intersector::WINDOW, ea.getX(), ea.getY() );

    osgUtil::IntersectionVisitor iv( intersector );

    osg::Camera* camera = viewer->getCamera();
    if( !camera )
      return false;

    camera->accept( iv );

    if( !intersector->containsIntersections() )
      return false;

    auto intersections = intersector->getIntersections();

    std::cout << "Got " << intersections.size() << " intersections:\n";

    for( auto&& intersection : intersections )
      std::cout << "  - Local intersection point = " << intersection.localIntersectionPoint << "\n";
  }

  return true;
}
