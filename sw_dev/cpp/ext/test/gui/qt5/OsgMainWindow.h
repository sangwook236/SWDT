#ifndef OsgMainWindow_h__
#define OsgMainWindow_h__

#include <QMainWindow>
#include <QMdiArea>

class OsgMainWindow : public QMainWindow
{
  Q_OBJECT

public:
  OsgMainWindow( QWidget* parent = 0, Qt::WindowFlags flags = 0 );
  ~OsgMainWindow();

private slots:
  void onCreateView();

private:
  QMdiArea* mdiArea_;
};

#endif
