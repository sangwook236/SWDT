#pragma once

#if !defined(__MY_WIDGET__H_)
#define __MY_WIDGET__H_ 1


#include <QDialog>
#include "ui_MyWidget.h"


class MyWidget: public QDialog, private Ui::MyWidget
{
	Q_OBJECT

public:
	MyWidget(QWidget *parent = nullptr);

protected slots:
	void sum();
	void average();

private:
	double sum_;
	double average_;
};


#endif  // __MY_WIDGET__H_
