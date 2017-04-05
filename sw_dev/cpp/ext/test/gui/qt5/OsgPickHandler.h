#ifndef OsgPickHandler_h__
#define OsgPickHandler_h__

#include <osgGA/GUIEventHandler>

class OsgPickHandler : public osgGA::GUIEventHandler
{
public:
  virtual ~OsgPickHandler();

  virtual bool handle( const osgGA::GUIEventAdapter&  ea,
                             osgGA::GUIActionAdapter& aa );

};

#endif
