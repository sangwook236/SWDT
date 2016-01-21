#include "MyWindow.h"
#include <QFont>
#include <QPushButton>
#include <iostream>


MyWindow::MyWindow(QWidget *parent /*= NULL*/)
: QWidget(parent), button_(NULL), area_(0)
{
    // Set size of the window.
    setFixedSize(400, 300);
    setWindowTitle("Qt4 Application");

    // Create and position the button.
    button_ = new QPushButton("Quit", this);
    button_->setFont(QFont("Times", 18, QFont::Bold));
    button_->setGeometry(10, 10, 80, 30);
    button_->resize(160, 30);

    QObject::connect(button_, SIGNAL(clicked()), this, SLOT(handleQuitButtonClicked()));
}

/*virtual*/ MyWindow::~MyWindow()
{
    delete button_;
    button_ = NULL;
}

void MyWindow::setArea(qreal area)
{
    if (area == area_) return;
    area_ = area;
    emit areaChanged();
}

void MyWindow::handAreaChanged()
{
    std::cout << "area is changed." << std::endl;
}

void MyWindow::handleQuitButtonClicked()
{
    emit quitButtonClicked();
}
