#include "MyMainWindow.h"
#include <QVBoxLayout>
#include <QListWidget>
#include <QPushButton>
#include <QTextEdit>
#include <QStatusBar>
#include <QToolBar>
#include <QMenu>
#include <QMenuBar>
#include <QFont>
#include <iostream>


MyMainWindow::MyMainWindow(QWidget *parent /*= NULL*/)
: QMainWindow(parent), button_(NULL), area_(0)
{
    // Set size of the window.
    {
		//setFixedSize(400, 300);
		resize(400,300);
		setWindowTitle("Qt4 Application");
    }

    // Create menu.
	{
		QPixmap newPix("new.png");
		QPixmap openPix("open.png");
		QPixmap quitPix("quit.png");

		QAction* newFile = new QAction(newPix, "&New...", this);
		QAction* openFile = new QAction(openPix, "&Open...", this);
		QAction* quit = new QAction(quitPix, "&Quit", this);

		newFile->setShortcut(tr("CTRL+N"));
		openFile->setShortcut(tr("CTRL+O"));
		quit->setShortcut(tr("CTRL+Q"));

		QMenu* menuFile = menuBar()->addMenu("&File");
		menuFile->addAction(newFile);
		menuFile->addAction(openFile);
		menuFile->addSeparator();
		menuFile->addAction(quit);

		connect(quit, SIGNAL(triggered()), this, SLOT(handleQuitMenuClicked()));

		//
		QAction* undo = new QAction("&Undo", this);
		QAction* redo = new QAction("&Redo", this);
		QAction* copy = new QAction("&Copy", this);
		QAction* cut = new QAction("Cut", this);
		QAction* paste = new QAction("&Paste", this);
		QAction* deleteItem = new QAction("&Delete", this);

		undo->setShortcut(tr("CTRL+Z"));
		redo->setShortcut(tr("CTRL+Y"));
		copy->setShortcut(tr("CTRL+C"));
		cut->setShortcut(tr("CTRL+X"));
		paste->setShortcut(tr("CTRL+V"));
		deleteItem->setShortcut(tr("CTRL+D"));

		QMenu* menuEdit = menuBar()->addMenu("&Edit");
		menuEdit->addAction(undo);
		menuEdit->addAction(redo);
		menuEdit->addSeparator();
		menuEdit->addAction(copy);
		menuEdit->addAction(cut);
		menuEdit->addAction(paste);
		menuEdit->addAction(deleteItem);

		//
		QAction* about = new QAction("&About...", this);

		QMenu* menuHelp = menuBar()->addMenu("&Help");
		menuHelp->addAction(about);
	}

	// Create toolbar.
	{
		QPixmap newPix("new.png");
		QPixmap openPix("open.png");
		QPixmap quitPix("quit.png");

		QToolBar* toolbar = addToolBar("main toolbar");
		toolbar->addAction(QIcon(newPix), "New File");
		toolbar->addAction(QIcon(openPix), "Open File");
		toolbar->addSeparator();
		QAction* quit = toolbar->addAction(QIcon(quitPix), "Quit Application");

		connect(quit, SIGNAL(triggered()), this, SLOT(handleQuitMenuClicked()));
	}

	// Create statusbar.
	{
		statusBar()->showMessage("Ready");
	}

	//
	{
		QVBoxLayout* vbox = new QVBoxLayout();
		QHBoxLayout* hbox = new QHBoxLayout();

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

    // Create and position a button.
    {
        button_ = new QPushButton("Quit", this);
        button_->setFont(QFont("Times", 18, QFont::Bold));
        button_->setGeometry(10, 10, 80, 30);
        button_->resize(160, 30);

        QObject::connect(button_, SIGNAL(clicked()), this, SLOT(handleQuitButtonClicked()));
    }
*/
}

/*virtual*/ MyMainWindow::~MyMainWindow()
{
}

void MyMainWindow::handleQuitButtonClicked()
{
    emit applicationQuitSignaled();
}

void MyMainWindow::handleQuitMenuClicked()
{
    emit applicationQuitSignaled();
}
