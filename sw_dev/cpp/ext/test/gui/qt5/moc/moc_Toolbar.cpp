/****************************************************************************
** Meta object code from reading C++ file 'Toolbar.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.8.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../Toolbar.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'Toolbar.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.8.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_ToolBar_t {
    QByteArrayData data[20];
    char stringdata0[190];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ToolBar_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ToolBar_t qt_meta_stringdata_ToolBar = {
    {
QT_MOC_LITERAL(0, 0, 7), // "ToolBar"
QT_MOC_LITERAL(1, 8, 5), // "order"
QT_MOC_LITERAL(2, 14, 0), // ""
QT_MOC_LITERAL(3, 15, 9), // "randomize"
QT_MOC_LITERAL(4, 25, 10), // "addSpinBox"
QT_MOC_LITERAL(5, 36, 13), // "removeSpinBox"
QT_MOC_LITERAL(6, 50, 13), // "changeMovable"
QT_MOC_LITERAL(7, 64, 7), // "movable"
QT_MOC_LITERAL(8, 72, 9), // "allowLeft"
QT_MOC_LITERAL(9, 82, 1), // "a"
QT_MOC_LITERAL(10, 84, 10), // "allowRight"
QT_MOC_LITERAL(11, 95, 8), // "allowTop"
QT_MOC_LITERAL(12, 104, 11), // "allowBottom"
QT_MOC_LITERAL(13, 116, 9), // "placeLeft"
QT_MOC_LITERAL(14, 126, 1), // "p"
QT_MOC_LITERAL(15, 128, 10), // "placeRight"
QT_MOC_LITERAL(16, 139, 8), // "placeTop"
QT_MOC_LITERAL(17, 148, 11), // "placeBottom"
QT_MOC_LITERAL(18, 160, 10), // "updateMenu"
QT_MOC_LITERAL(19, 171, 18) // "insertToolBarBreak"

    },
    "ToolBar\0order\0\0randomize\0addSpinBox\0"
    "removeSpinBox\0changeMovable\0movable\0"
    "allowLeft\0a\0allowRight\0allowTop\0"
    "allowBottom\0placeLeft\0p\0placeRight\0"
    "placeTop\0placeBottom\0updateMenu\0"
    "insertToolBarBreak"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ToolBar[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      15,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   89,    2, 0x08 /* Private */,
       3,    0,   90,    2, 0x08 /* Private */,
       4,    0,   91,    2, 0x08 /* Private */,
       5,    0,   92,    2, 0x08 /* Private */,
       6,    1,   93,    2, 0x08 /* Private */,
       8,    1,   96,    2, 0x08 /* Private */,
      10,    1,   99,    2, 0x08 /* Private */,
      11,    1,  102,    2, 0x08 /* Private */,
      12,    1,  105,    2, 0x08 /* Private */,
      13,    1,  108,    2, 0x08 /* Private */,
      15,    1,  111,    2, 0x08 /* Private */,
      16,    1,  114,    2, 0x08 /* Private */,
      17,    1,  117,    2, 0x08 /* Private */,
      18,    0,  120,    2, 0x08 /* Private */,
      19,    0,  121,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,    7,
    QMetaType::Void, QMetaType::Bool,    9,
    QMetaType::Void, QMetaType::Bool,    9,
    QMetaType::Void, QMetaType::Bool,    9,
    QMetaType::Void, QMetaType::Bool,    9,
    QMetaType::Void, QMetaType::Bool,   14,
    QMetaType::Void, QMetaType::Bool,   14,
    QMetaType::Void, QMetaType::Bool,   14,
    QMetaType::Void, QMetaType::Bool,   14,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void ToolBar::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        ToolBar *_t = static_cast<ToolBar *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->order(); break;
        case 1: _t->randomize(); break;
        case 2: _t->addSpinBox(); break;
        case 3: _t->removeSpinBox(); break;
        case 4: _t->changeMovable((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 5: _t->allowLeft((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 6: _t->allowRight((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 7: _t->allowTop((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 8: _t->allowBottom((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 9: _t->placeLeft((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 10: _t->placeRight((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 11: _t->placeTop((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 12: _t->placeBottom((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 13: _t->updateMenu(); break;
        case 14: _t->insertToolBarBreak(); break;
        default: ;
        }
    }
}

const QMetaObject ToolBar::staticMetaObject = {
    { &QToolBar::staticMetaObject, qt_meta_stringdata_ToolBar.data,
      qt_meta_data_ToolBar,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *ToolBar::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ToolBar::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_ToolBar.stringdata0))
        return static_cast<void*>(const_cast< ToolBar*>(this));
    return QToolBar::qt_metacast(_clname);
}

int ToolBar::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QToolBar::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 15)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 15;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 15)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 15;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
