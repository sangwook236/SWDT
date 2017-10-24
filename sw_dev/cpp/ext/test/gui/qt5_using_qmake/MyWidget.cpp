#include <QtGui>
#include "MyWidget.h"


MyWidget::MyWidget(QWidget *parent /*= nullptr*/)
: QDialog(parent)
{
	setupUi(this);

	connect(pushButtonSum, SIGNAL(clicked()), this, SLOT(sum()));
	connect(pushButtonAverage, SIGNAL(clicked()), this, SLOT(average()));
}

void MyWidget::sum()
{
	sum_ = lineEditKorean->text().toInt();
	sum_ += lineEditEnglish->text().toInt();
	sum_ += lineEditMath->text().toInt();

	labelSum->setText(QString("%1").arg(sum_));
}

void MyWidget::average()
{
	sum();

	average_ = sum_ / 3.0;
	labelAverage->setText(QString("%1").arg(average_));
}
