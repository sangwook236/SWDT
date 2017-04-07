#ifndef __MY_SLOT_CLASS_H_
#define __MY_SLOT_CLASS_H_


#include <QObject>
#include <QQuickItem>
#include <QDebug>

QT_FORWARD_DECLARE_CLASS(QString)

class MySlotClass : public QObject
{
	Q_OBJECT
public slots:
	void cppSlot(const QString &msg)
	{
		qDebug() << "Called the C++ slot with message:" << msg;
	}
	void cppSlotObject(const QVariant &obj)
	{
		qDebug() << "Called the C++ slot with object:" << obj;

		QQuickItem *item = qobject_cast<QQuickItem *>(obj.value<QObject *>());
		qDebug() << "Item dimensions:" << item->width() << item->height();
	}
};


#endif // __MY_SLOT_CLASS_H_
