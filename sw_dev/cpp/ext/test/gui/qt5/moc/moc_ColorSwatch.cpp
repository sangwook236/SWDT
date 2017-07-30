/****************************************************************************
** Meta object code from reading C++ file 'ColorSwatch.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.8.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../ColorSwatch.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ColorSwatch.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.8.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_ColorSwatch_t {
    QByteArrayData data[23];
    char stringdata0[239];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ColorSwatch_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ColorSwatch_t qt_meta_stringdata_ColorSwatch = {
    {
QT_MOC_LITERAL(0, 0, 11), // "ColorSwatch"
QT_MOC_LITERAL(1, 12, 14), // "changeClosable"
QT_MOC_LITERAL(2, 27, 0), // ""
QT_MOC_LITERAL(3, 28, 2), // "on"
QT_MOC_LITERAL(4, 31, 13), // "changeMovable"
QT_MOC_LITERAL(5, 45, 15), // "changeFloatable"
QT_MOC_LITERAL(6, 61, 14), // "changeFloating"
QT_MOC_LITERAL(7, 76, 22), // "changeVerticalTitleBar"
QT_MOC_LITERAL(8, 99, 17), // "updateContextMenu"
QT_MOC_LITERAL(9, 117, 9), // "allowLeft"
QT_MOC_LITERAL(10, 127, 1), // "a"
QT_MOC_LITERAL(11, 129, 10), // "allowRight"
QT_MOC_LITERAL(12, 140, 8), // "allowTop"
QT_MOC_LITERAL(13, 149, 11), // "allowBottom"
QT_MOC_LITERAL(14, 161, 9), // "placeLeft"
QT_MOC_LITERAL(15, 171, 1), // "p"
QT_MOC_LITERAL(16, 173, 10), // "placeRight"
QT_MOC_LITERAL(17, 184, 8), // "placeTop"
QT_MOC_LITERAL(18, 193, 11), // "placeBottom"
QT_MOC_LITERAL(19, 205, 9), // "splitInto"
QT_MOC_LITERAL(20, 215, 8), // "QAction*"
QT_MOC_LITERAL(21, 224, 6), // "action"
QT_MOC_LITERAL(22, 231, 7) // "tabInto"

    },
    "ColorSwatch\0changeClosable\0\0on\0"
    "changeMovable\0changeFloatable\0"
    "changeFloating\0changeVerticalTitleBar\0"
    "updateContextMenu\0allowLeft\0a\0allowRight\0"
    "allowTop\0allowBottom\0placeLeft\0p\0"
    "placeRight\0placeTop\0placeBottom\0"
    "splitInto\0QAction*\0action\0tabInto"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ColorSwatch[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      16,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   94,    2, 0x08 /* Private */,
       4,    1,   97,    2, 0x08 /* Private */,
       5,    1,  100,    2, 0x08 /* Private */,
       6,    1,  103,    2, 0x08 /* Private */,
       7,    1,  106,    2, 0x08 /* Private */,
       8,    0,  109,    2, 0x08 /* Private */,
       9,    1,  110,    2, 0x08 /* Private */,
      11,    1,  113,    2, 0x08 /* Private */,
      12,    1,  116,    2, 0x08 /* Private */,
      13,    1,  119,    2, 0x08 /* Private */,
      14,    1,  122,    2, 0x08 /* Private */,
      16,    1,  125,    2, 0x08 /* Private */,
      17,    1,  128,    2, 0x08 /* Private */,
      18,    1,  131,    2, 0x08 /* Private */,
      19,    1,  134,    2, 0x08 /* Private */,
      22,    1,  137,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void, QMetaType::Bool,    3,
    QMetaType::Void, QMetaType::Bool,    3,
    QMetaType::Void, QMetaType::Bool,    3,
    QMetaType::Void, QMetaType::Bool,    3,
    QMetaType::Void, QMetaType::Bool,    3,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Bool,   10,
    QMetaType::Void, QMetaType::Bool,   10,
    QMetaType::Void, QMetaType::Bool,   10,
    QMetaType::Void, QMetaType::Bool,   10,
    QMetaType::Void, QMetaType::Bool,   15,
    QMetaType::Void, QMetaType::Bool,   15,
    QMetaType::Void, QMetaType::Bool,   15,
    QMetaType::Void, QMetaType::Bool,   15,
    QMetaType::Void, 0x80000000 | 20,   21,
    QMetaType::Void, 0x80000000 | 20,   21,

       0        // eod
};

void ColorSwatch::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        ColorSwatch *_t = static_cast<ColorSwatch *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->changeClosable((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 1: _t->changeMovable((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 2: _t->changeFloatable((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 3: _t->changeFloating((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 4: _t->changeVerticalTitleBar((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 5: _t->updateContextMenu(); break;
        case 6: _t->allowLeft((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 7: _t->allowRight((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 8: _t->allowTop((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 9: _t->allowBottom((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 10: _t->placeLeft((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 11: _t->placeRight((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 12: _t->placeTop((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 13: _t->placeBottom((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 14: _t->splitInto((*reinterpret_cast< QAction*(*)>(_a[1]))); break;
        case 15: _t->tabInto((*reinterpret_cast< QAction*(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObject ColorSwatch::staticMetaObject = {
    { &QDockWidget::staticMetaObject, qt_meta_stringdata_ColorSwatch.data,
      qt_meta_data_ColorSwatch,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *ColorSwatch::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ColorSwatch::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_ColorSwatch.stringdata0))
        return static_cast<void*>(const_cast< ColorSwatch*>(this));
    return QDockWidget::qt_metacast(_clname);
}

int ColorSwatch::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QDockWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 16)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 16;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 16)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 16;
    }
    return _id;
}
struct qt_meta_stringdata_BlueTitleBar_t {
    QByteArrayData data[3];
    char stringdata0[25];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_BlueTitleBar_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_BlueTitleBar_t qt_meta_stringdata_BlueTitleBar = {
    {
QT_MOC_LITERAL(0, 0, 12), // "BlueTitleBar"
QT_MOC_LITERAL(1, 13, 10), // "updateMask"
QT_MOC_LITERAL(2, 24, 0) // ""

    },
    "BlueTitleBar\0updateMask\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_BlueTitleBar[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   19,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void,

       0        // eod
};

void BlueTitleBar::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        BlueTitleBar *_t = static_cast<BlueTitleBar *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->updateMask(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObject BlueTitleBar::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_BlueTitleBar.data,
      qt_meta_data_BlueTitleBar,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *BlueTitleBar::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *BlueTitleBar::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_BlueTitleBar.stringdata0))
        return static_cast<void*>(const_cast< BlueTitleBar*>(this));
    return QWidget::qt_metacast(_clname);
}

int BlueTitleBar::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 1)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 1;
    }
    return _id;
}
struct qt_meta_stringdata_ColorDock_t {
    QByteArrayData data[3];
    char stringdata0[27];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_ColorDock_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_ColorDock_t qt_meta_stringdata_ColorDock = {
    {
QT_MOC_LITERAL(0, 0, 9), // "ColorDock"
QT_MOC_LITERAL(1, 10, 15), // "changeSizeHints"
QT_MOC_LITERAL(2, 26, 0) // ""

    },
    "ColorDock\0changeSizeHints\0"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_ColorDock[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   19,    2, 0x0a /* Public */,

 // slots: parameters
    QMetaType::Void,

       0        // eod
};

void ColorDock::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        ColorDock *_t = static_cast<ColorDock *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->changeSizeHints(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObject ColorDock::staticMetaObject = {
    { &QFrame::staticMetaObject, qt_meta_stringdata_ColorDock.data,
      qt_meta_data_ColorDock,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *ColorDock::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *ColorDock::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_ColorDock.stringdata0))
        return static_cast<void*>(const_cast< ColorDock*>(this));
    return QFrame::qt_metacast(_clname);
}

int ColorDock::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QFrame::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 1)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 1;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
