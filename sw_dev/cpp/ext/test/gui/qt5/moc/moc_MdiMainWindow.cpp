/****************************************************************************
** Meta object code from reading C++ file 'MdiMainWindow.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.6.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../MdiMainWindow.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'MdiMainWindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.6.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_MdiMainWindow_t {
    QByteArrayData data[17];
    char stringdata0[176];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_MdiMainWindow_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_MdiMainWindow_t qt_meta_stringdata_MdiMainWindow = {
    {
QT_MOC_LITERAL(0, 0, 13), // "MdiMainWindow"
QT_MOC_LITERAL(1, 14, 7), // "newFile"
QT_MOC_LITERAL(2, 22, 0), // ""
QT_MOC_LITERAL(3, 23, 4), // "open"
QT_MOC_LITERAL(4, 28, 4), // "save"
QT_MOC_LITERAL(5, 33, 6), // "saveAs"
QT_MOC_LITERAL(6, 40, 23), // "updateRecentFileActions"
QT_MOC_LITERAL(7, 64, 14), // "openRecentFile"
QT_MOC_LITERAL(8, 79, 3), // "cut"
QT_MOC_LITERAL(9, 83, 4), // "copy"
QT_MOC_LITERAL(10, 88, 5), // "paste"
QT_MOC_LITERAL(11, 94, 5), // "about"
QT_MOC_LITERAL(12, 100, 11), // "updateMenus"
QT_MOC_LITERAL(13, 112, 16), // "updateWindowMenu"
QT_MOC_LITERAL(14, 129, 14), // "createMdiChild"
QT_MOC_LITERAL(15, 144, 9), // "MdiChild*"
QT_MOC_LITERAL(16, 154, 21) // "switchLayoutDirection"

    },
    "MdiMainWindow\0newFile\0\0open\0save\0"
    "saveAs\0updateRecentFileActions\0"
    "openRecentFile\0cut\0copy\0paste\0about\0"
    "updateMenus\0updateWindowMenu\0"
    "createMdiChild\0MdiChild*\0switchLayoutDirection"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MdiMainWindow[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      14,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   84,    2, 0x08 /* Private */,
       3,    0,   85,    2, 0x08 /* Private */,
       4,    0,   86,    2, 0x08 /* Private */,
       5,    0,   87,    2, 0x08 /* Private */,
       6,    0,   88,    2, 0x08 /* Private */,
       7,    0,   89,    2, 0x08 /* Private */,
       8,    0,   90,    2, 0x08 /* Private */,
       9,    0,   91,    2, 0x08 /* Private */,
      10,    0,   92,    2, 0x08 /* Private */,
      11,    0,   93,    2, 0x08 /* Private */,
      12,    0,   94,    2, 0x08 /* Private */,
      13,    0,   95,    2, 0x08 /* Private */,
      14,    0,   96,    2, 0x08 /* Private */,
      16,    0,   97,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    0x80000000 | 15,
    QMetaType::Void,

       0        // eod
};

void MdiMainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        MdiMainWindow *_t = static_cast<MdiMainWindow *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->newFile(); break;
        case 1: _t->open(); break;
        case 2: _t->save(); break;
        case 3: _t->saveAs(); break;
        case 4: _t->updateRecentFileActions(); break;
        case 5: _t->openRecentFile(); break;
        case 6: _t->cut(); break;
        case 7: _t->copy(); break;
        case 8: _t->paste(); break;
        case 9: _t->about(); break;
        case 10: _t->updateMenus(); break;
        case 11: _t->updateWindowMenu(); break;
        case 12: { MdiChild* _r = _t->createMdiChild();
            if (_a[0]) *reinterpret_cast< MdiChild**>(_a[0]) = _r; }  break;
        case 13: _t->switchLayoutDirection(); break;
        default: ;
        }
    }
}

const QMetaObject MdiMainWindow::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_MdiMainWindow.data,
      qt_meta_data_MdiMainWindow,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *MdiMainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MdiMainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_MdiMainWindow.stringdata0))
        return static_cast<void*>(const_cast< MdiMainWindow*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int MdiMainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 14)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 14;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 14)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 14;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
