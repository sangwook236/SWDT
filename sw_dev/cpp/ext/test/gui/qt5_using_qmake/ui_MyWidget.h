/********************************************************************************
** Form generated from reading UI file 'MyWidget.ui'
**
** Created by: Qt User Interface Compiler version 5.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MYWIDGET_H
#define UI_MYWIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MyWidget
{
public:
    QLabel *label;
    QLabel *label_2;
    QLabel *label_3;
    QLabel *label_4;
    QLineEdit *lineEditName;
    QLineEdit *lineEditKorean;
    QLineEdit *lineEditEnglish;
    QLineEdit *lineEditMath;
    QPushButton *pushButtonSum;
    QPushButton *pushButtonAverage;
    QLabel *labelSum;
    QLabel *labelAverage;

    void setupUi(QWidget *MyWidget)
    {
        if (MyWidget->objectName().isEmpty())
            MyWidget->setObjectName(QStringLiteral("MyWidget"));
        MyWidget->resize(371, 322);
        label = new QLabel(MyWidget);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(90, 70, 61, 21));
        label_2 = new QLabel(MyWidget);
        label_2->setObjectName(QStringLiteral("label_2"));
        label_2->setGeometry(QRect(90, 100, 61, 21));
        label_3 = new QLabel(MyWidget);
        label_3->setObjectName(QStringLiteral("label_3"));
        label_3->setGeometry(QRect(90, 130, 61, 21));
        label_4 = new QLabel(MyWidget);
        label_4->setObjectName(QStringLiteral("label_4"));
        label_4->setGeometry(QRect(90, 160, 61, 21));
        lineEditName = new QLineEdit(MyWidget);
        lineEditName->setObjectName(QStringLiteral("lineEditName"));
        lineEditName->setGeometry(QRect(150, 70, 113, 21));
        lineEditKorean = new QLineEdit(MyWidget);
        lineEditKorean->setObjectName(QStringLiteral("lineEditKorean"));
        lineEditKorean->setGeometry(QRect(150, 100, 113, 21));
        lineEditEnglish = new QLineEdit(MyWidget);
        lineEditEnglish->setObjectName(QStringLiteral("lineEditEnglish"));
        lineEditEnglish->setGeometry(QRect(150, 130, 113, 21));
        lineEditMath = new QLineEdit(MyWidget);
        lineEditMath->setObjectName(QStringLiteral("lineEditMath"));
        lineEditMath->setGeometry(QRect(150, 160, 113, 21));
        pushButtonSum = new QPushButton(MyWidget);
        pushButtonSum->setObjectName(QStringLiteral("pushButtonSum"));
        pushButtonSum->setGeometry(QRect(90, 190, 93, 28));
        pushButtonAverage = new QPushButton(MyWidget);
        pushButtonAverage->setObjectName(QStringLiteral("pushButtonAverage"));
        pushButtonAverage->setGeometry(QRect(90, 220, 93, 28));
        labelSum = new QLabel(MyWidget);
        labelSum->setObjectName(QStringLiteral("labelSum"));
        labelSum->setGeometry(QRect(200, 190, 91, 21));
        labelAverage = new QLabel(MyWidget);
        labelAverage->setObjectName(QStringLiteral("labelAverage"));
        labelAverage->setGeometry(QRect(200, 220, 91, 21));

        retranslateUi(MyWidget);

        QMetaObject::connectSlotsByName(MyWidget);
    } // setupUi

    void retranslateUi(QWidget *MyWidget)
    {
        MyWidget->setWindowTitle(QApplication::translate("MyWidget", "Form", Q_NULLPTR));
        label->setText(QApplication::translate("MyWidget", "Name:", Q_NULLPTR));
        label_2->setText(QApplication::translate("MyWidget", "Korean:", Q_NULLPTR));
        label_3->setText(QApplication::translate("MyWidget", "English:", Q_NULLPTR));
        label_4->setText(QApplication::translate("MyWidget", "Math:", Q_NULLPTR));
        pushButtonSum->setText(QApplication::translate("MyWidget", "Sum", Q_NULLPTR));
        pushButtonAverage->setText(QApplication::translate("MyWidget", "Average", Q_NULLPTR));
        labelSum->setText(QString());
        labelAverage->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class MyWidget: public Ui_MyWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MYWIDGET_H
