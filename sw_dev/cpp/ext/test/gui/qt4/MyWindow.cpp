#include "MyWindow.h"
#include <QVBoxLayout>
#include <QListWidget>
#include <QPushButton>
#include <iostream>


MyWindow::MyWindow(QWidget *parent /*= NULL*/)
: QWidget(parent), area_(0)
{
    // Set size of the window.
    {
		//setFixedSize(400, 300);
		resize(400,300);
		setWindowTitle("Qt4 Application #2");
    }

	//
	{

		QVBoxLayout* vbox = new QVBoxLayout();
		QHBoxLayout* hbox = new QHBoxLayout(this);

		QListWidget* lw = new QListWidget(this);
		lw->addItem("The Omen");
		lw->addItem("The Exorcist");
		lw->addItem("Notes on a scandal");
		lw->addItem("Fargo");
		lw->addItem("Capote");

		QPushButton *add = new QPushButton("Add", this);
		QPushButton *rename = new QPushButton("Rename", this);
		QPushButton *remove = new QPushButton("Remove", this);
		QPushButton *removeall = new QPushButton("Remove All", this);

		vbox->setSpacing(3);
		vbox->addStretch(1);
		vbox->addWidget(add);
		vbox->addWidget(rename);
		vbox->addWidget(remove);
		vbox->addWidget(removeall);
		vbox->addStretch(1);

		hbox->addWidget(lw);
		hbox->addSpacing(15);
		hbox->addLayout(vbox);

		setLayout(hbox);
	}
/*
	{
		QTextEdit* edit = new QTextEdit(this);

		setCentralWidget(edit);
	}
*/
}

/*virtual*/ MyWindow::~MyWindow()
{
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
    emit applicationQuitSignaled();
}

void MyWindow::handleQuitMenuClicked()
{
    emit applicationQuitSignaled();
}
