/****************************************************************************
**
** Copyright (C) 2015 The Qt Company Ltd.
** Contact: http://www.qt.io/licensing/
**
** This file is part of the demonstration applications of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:LGPL21$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see http://www.qt.io/terms-conditions. For further
** information use the contact form at http://www.qt.io/contact-us.
**
** GNU Lesser General Public License Usage
** Alternatively, this file may be used under the terms of the GNU Lesser
** General Public License version 2.1 or version 3 as published by the Free
** Software Foundation and appearing in the file LICENSE.LGPLv21 and
** LICENSE.LGPLv3 included in the packaging of this file. Please review the
** following information to ensure the GNU Lesser General Public License
** requirements will be met: https://www.gnu.org/licenses/lgpl.html and
** http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
**
** As a special exception, The Qt Company gives you certain additional
** rights. These rights are described in The Qt Company LGPL Exception
** version 1.1, included in the file LGPL_EXCEPTION.txt in this package.
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include "Toolbar.h"

#include <QMainWindow>
#include <QMenu>
#include <QPainter>
#include <QPainterPath>
#include <QSpinBox>
#include <QLabel>
#include <QToolTip>

#include <stdlib.h>

//--S [] 2017/04/05: Sang-Wook Lee.
namespace my_qt5 {

void render_qt_text(QPainter *, int, int, const QColor &);

}  // namespace my_qt5
//--E [] 2017/04/05: Sang-Wook Lee.

static QPixmap genIcon(const QSize &iconSize, const QString &, const QColor &color)
{
    int w = iconSize.width();
    int h = iconSize.height();

    QImage image(w, h, QImage::Format_ARGB32_Premultiplied);
    image.fill(0);

    QPainter p(&image);

	//--S [] 2017/04/05: Sang-Wook Lee.
	//extern void render_qt_text(QPainter *, int, int, const QColor &);
	//render_qt_text(&p, w, h, color);
	my_qt5::render_qt_text(&p, w, h, color);
	//--E [] 2017/04/05: Sang-Wook Lee.

    return QPixmap::fromImage(image, Qt::DiffuseDither | Qt::DiffuseAlphaDither);
}

static QPixmap genIcon(const QSize &iconSize, int number, const QColor &color)
{ return genIcon(iconSize, QString::number(number), color); }

ToolBar::ToolBar(const QString &title, QWidget *parent)
    : QToolBar(parent)
    , spinbox(Q_NULLPTR)
    , spinboxAction(Q_NULLPTR)
{
    setWindowTitle(title);
    setObjectName(title);

    setIconSize(QSize(32, 32));

    menu = new QMenu("One", this);
    menu->setIcon(genIcon(iconSize(), 1, Qt::black));
    menu->addAction(genIcon(iconSize(), "A", Qt::blue), "A");
    menu->addAction(genIcon(iconSize(), "B", Qt::blue), "B");
    menu->addAction(genIcon(iconSize(), "C", Qt::blue), "C");
    addAction(menu->menuAction());

    QAction *two = addAction(genIcon(iconSize(), 2, Qt::white), "Two");
    QFont boldFont;
    boldFont.setBold(true);
    two->setFont(boldFont);

    addAction(genIcon(iconSize(), 3, Qt::red), "Three");
    addAction(genIcon(iconSize(), 4, Qt::green), "Four");
    addAction(genIcon(iconSize(), 5, Qt::blue), "Five");
    addAction(genIcon(iconSize(), 6, Qt::yellow), "Six");
    orderAction = new QAction(this);
    orderAction->setText(tr("Order Items in Tool Bar"));
    connect(orderAction, &QAction::triggered, this, &ToolBar::order);

    randomizeAction = new QAction(this);
    randomizeAction->setText(tr("Randomize Items in Tool Bar"));
    connect(randomizeAction, &QAction::triggered, this, &ToolBar::randomize);

    addSpinBoxAction = new QAction(this);
    addSpinBoxAction->setText(tr("Add Spin Box"));
    connect(addSpinBoxAction, &QAction::triggered, this, &ToolBar::addSpinBox);

    removeSpinBoxAction = new QAction(this);
    removeSpinBoxAction->setText(tr("Remove Spin Box"));
    removeSpinBoxAction->setEnabled(false);
    connect(removeSpinBoxAction, &QAction::triggered, this, &ToolBar::removeSpinBox);

    movableAction = new QAction(tr("Movable"), this);
    movableAction->setCheckable(true);
    connect(movableAction, &QAction::triggered, this, &ToolBar::changeMovable);

    allowedAreasActions = new QActionGroup(this);
    allowedAreasActions->setExclusive(false);

    allowLeftAction = new QAction(tr("Allow on Left"), this);
    allowLeftAction->setCheckable(true);
    connect(allowLeftAction, &QAction::triggered, this, &ToolBar::allowLeft);

    allowRightAction = new QAction(tr("Allow on Right"), this);
    allowRightAction->setCheckable(true);
    connect(allowRightAction, &QAction::triggered, this, &ToolBar::allowRight);

    allowTopAction = new QAction(tr("Allow on Top"), this);
    allowTopAction->setCheckable(true);
    connect(allowTopAction, &QAction::triggered, this, &ToolBar::allowTop);

    allowBottomAction = new QAction(tr("Allow on Bottom"), this);
    allowBottomAction->setCheckable(true);
    connect(allowBottomAction, &QAction::triggered, this, &ToolBar::allowBottom);

    allowedAreasActions->addAction(allowLeftAction);
    allowedAreasActions->addAction(allowRightAction);
    allowedAreasActions->addAction(allowTopAction);
    allowedAreasActions->addAction(allowBottomAction);

    areaActions = new QActionGroup(this);
    areaActions->setExclusive(true);

    leftAction = new QAction(tr("Place on Left") , this);
    leftAction->setCheckable(true);
    connect(leftAction, &QAction::triggered, this, &ToolBar::placeLeft);

    rightAction = new QAction(tr("Place on Right") , this);
    rightAction->setCheckable(true);
    connect(rightAction, &QAction::triggered, this, &ToolBar::placeRight);

    topAction = new QAction(tr("Place on Top") , this);
    topAction->setCheckable(true);
    connect(topAction, &QAction::triggered, this, &ToolBar::placeTop);

    bottomAction = new QAction(tr("Place on Bottom") , this);
    bottomAction->setCheckable(true);
    connect(bottomAction, &QAction::triggered, this, &ToolBar::placeBottom);

    areaActions->addAction(leftAction);
    areaActions->addAction(rightAction);
    areaActions->addAction(topAction);
    areaActions->addAction(bottomAction);

    connect(movableAction, &QAction::triggered, areaActions, &QActionGroup::setEnabled);

    connect(movableAction, &QAction::triggered, allowedAreasActions, &QActionGroup::setEnabled);

    menu = new QMenu(title, this);
    menu->addAction(toggleViewAction());
    menu->addSeparator();
    menu->addAction(orderAction);
    menu->addAction(randomizeAction);
    menu->addSeparator();
    menu->addAction(addSpinBoxAction);
    menu->addAction(removeSpinBoxAction);
    menu->addSeparator();
    menu->addAction(movableAction);
    menu->addSeparator();
    menu->addActions(allowedAreasActions->actions());
    menu->addSeparator();
    menu->addActions(areaActions->actions());
    menu->addSeparator();
    menu->addAction(tr("Insert break"), this, &ToolBar::insertToolBarBreak);

    connect(menu, &QMenu::aboutToShow, this, &ToolBar::updateMenu);

    randomize();
}

void ToolBar::updateMenu()
{
    QMainWindow *mainWindow = qobject_cast<QMainWindow *>(parentWidget());
    Q_ASSERT(mainWindow != 0);

    const Qt::ToolBarArea area = mainWindow->toolBarArea(this);
    const Qt::ToolBarAreas areas = allowedAreas();

    movableAction->setChecked(isMovable());

    allowLeftAction->setChecked(isAreaAllowed(Qt::LeftToolBarArea));
    allowRightAction->setChecked(isAreaAllowed(Qt::RightToolBarArea));
    allowTopAction->setChecked(isAreaAllowed(Qt::TopToolBarArea));
    allowBottomAction->setChecked(isAreaAllowed(Qt::BottomToolBarArea));

    if (allowedAreasActions->isEnabled()) {
        allowLeftAction->setEnabled(area != Qt::LeftToolBarArea);
        allowRightAction->setEnabled(area != Qt::RightToolBarArea);
        allowTopAction->setEnabled(area != Qt::TopToolBarArea);
        allowBottomAction->setEnabled(area != Qt::BottomToolBarArea);
    }

    leftAction->setChecked(area == Qt::LeftToolBarArea);
    rightAction->setChecked(area == Qt::RightToolBarArea);
    topAction->setChecked(area == Qt::TopToolBarArea);
    bottomAction->setChecked(area == Qt::BottomToolBarArea);

    if (areaActions->isEnabled()) {
        leftAction->setEnabled(areas & Qt::LeftToolBarArea);
        rightAction->setEnabled(areas & Qt::RightToolBarArea);
        topAction->setEnabled(areas & Qt::TopToolBarArea);
        bottomAction->setEnabled(areas & Qt::BottomToolBarArea);
    }
}

void ToolBar::order()
{
    QList<QAction *> ordered;
    QList<QAction *> actions1 = actions();
    foreach (QAction *action, findChildren<QAction *>()) {
        if (!actions1.contains(action))
            continue;
        actions1.removeAll(action);
        ordered.append(action);
    }

    clear();
    addActions(ordered);

    orderAction->setEnabled(false);
}

void ToolBar::randomize()
{
    QList<QAction *> randomized;
    QList<QAction *> actions = this->actions();
    while (!actions.isEmpty()) {
        QAction *action = actions.takeAt(rand() % actions.size());
        randomized.append(action);
    }
    clear();
    addActions(randomized);

    orderAction->setEnabled(true);
}

void ToolBar::addSpinBox()
{
    if (!spinbox)
        spinbox = new QSpinBox(this);
    if (!spinboxAction)
        spinboxAction = addWidget(spinbox);
    else
        addAction(spinboxAction);

    addSpinBoxAction->setEnabled(false);
    removeSpinBoxAction->setEnabled(true);
}

void ToolBar::removeSpinBox()
{
    if (spinboxAction)
        removeAction(spinboxAction);

    addSpinBoxAction->setEnabled(true);
    removeSpinBoxAction->setEnabled(false);
}

void ToolBar::allow(Qt::ToolBarArea area, bool a)
{
    Qt::ToolBarAreas areas = allowedAreas();
    areas = a ? areas | area : areas & ~area;
    setAllowedAreas(areas);

    if (areaActions->isEnabled()) {
        leftAction->setEnabled(areas & Qt::LeftToolBarArea);
        rightAction->setEnabled(areas & Qt::RightToolBarArea);
        topAction->setEnabled(areas & Qt::TopToolBarArea);
        bottomAction->setEnabled(areas & Qt::BottomToolBarArea);
    }
}

void ToolBar::place(Qt::ToolBarArea area, bool p)
{
    if (!p)
        return;

    QMainWindow *mainWindow = qobject_cast<QMainWindow *>(parentWidget());
    Q_ASSERT(mainWindow != 0);

    mainWindow->addToolBar(area, this);

    if (allowedAreasActions->isEnabled()) {
        allowLeftAction->setEnabled(area != Qt::LeftToolBarArea);
        allowRightAction->setEnabled(area != Qt::RightToolBarArea);
        allowTopAction->setEnabled(area != Qt::TopToolBarArea);
        allowBottomAction->setEnabled(area != Qt::BottomToolBarArea);
    }
}

void ToolBar::changeMovable(bool movable)
{ setMovable(movable); }

void ToolBar::allowLeft(bool a)
{ allow(Qt::LeftToolBarArea, a); }

void ToolBar::allowRight(bool a)
{ allow(Qt::RightToolBarArea, a); }

void ToolBar::allowTop(bool a)
{ allow(Qt::TopToolBarArea, a); }

void ToolBar::allowBottom(bool a)
{ allow(Qt::BottomToolBarArea, a); }

void ToolBar::placeLeft(bool p)
{ place(Qt::LeftToolBarArea, p); }

void ToolBar::placeRight(bool p)
{ place(Qt::RightToolBarArea, p); }

void ToolBar::placeTop(bool p)
{ place(Qt::TopToolBarArea, p); }

void ToolBar::placeBottom(bool p)
{ place(Qt::BottomToolBarArea, p); }

void ToolBar::insertToolBarBreak()
{
    QMainWindow *mainWindow = qobject_cast<QMainWindow *>(parentWidget());
    Q_ASSERT(mainWindow != 0);

    mainWindow->insertToolBarBreak(this);
}
