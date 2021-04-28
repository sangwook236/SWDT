#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget, QPushButton, QLineEdit, QBoxLayout
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QDrag, QMouseEvent, QPixmap
from PyQt5.QtCore import Qt, QMimeData, QPoint

# REF [site] >> https://github.com/RavenKyu/OpenTutorials_PyQt/blob/master/QtFramework/QtWidgets/Drag_and_Drop/drag_and_drop_00_basic.py
def drag_and_drop_text_example():
	class MyButton(QPushButton):
		def __init__(self, title):
			super().__init__(title)

			self.setAcceptDrops(True)

		def dragEnterEvent(self, event: QDragEnterEvent):
			if event.mimeData().hasFormat('text/plain'):
				event.accept()
			else:
				event.ignore()

		def dropEvent(self, event: QDropEvent):
			self.setText(event.mimeData().text())

	class MyForm(QWidget):
		def __init__(self, parent=None):
			super().__init__(parent, flags=Qt.Widget)

			self.setWindowTitle('Drag and Drop')
			self.init_ui()

		def init_ui(self):
			layout = QBoxLayout(QBoxLayout.TopToBottom, self)
			self.setLayout(layout)

			line_edit = QLineEdit()
			line_edit.setDragEnabled(True)
			layout.addWidget(line_edit)

			btn = MyButton('Button!')
			layout.addWidget(btn)

	#--------------------
	app = QApplication(sys.argv)

	form = MyForm()
	form.show()

	sys.exit(app.exec_())

# REF [site] >> https://github.com/RavenKyu/OpenTutorials_PyQt/blob/master/QtFramework/QtWidgets/Drag_and_Drop/drag_and_drop_01_move_button.py
def drag_and_drop_button_example1():
	class MyButton(QPushButton):
		def __init__(self, title, parent):
			super().__init__(title, parent)

		def mouseMoveEvent(self, event: QMouseEvent):
			if event.buttons() != Qt.RightButton:
				return

			drag = QDrag(self)

			mime_data = QMimeData()
			drag.setMimeData(mime_data)

			drag.exec_(Qt.MoveAction)

	class MyForm(QWidget):
		def __init__(self, parent=None):
			super().__init__(parent, flags=Qt.Widget)

			self.setWindowTitle('Drag and Drop')
			self.setFixedSize(300, 300)
			self.setAcceptDrops(True)

			self.init_ui()

		def init_ui(self):
			self.btn = MyButton('Drag me to the moon', self)
			self.btn.show()

		def dragEnterEvent(self, event: QDragEnterEvent):
			event.accept()

		def dropEvent(self, event: QDropEvent):
			position = event.pos()
			self.btn.move(position)

			event.setDropAction(Qt.MoveAction)
			event.accept()

	#--------------------
	app = QApplication(sys.argv)

	form = MyForm()
	form.show()

	sys.exit(app.exec_())

# REF [site] >> https://github.com/RavenKyu/OpenTutorials_PyQt/blob/master/QtFramework/QtWidgets/Drag_and_Drop/drag_and_drop_02_keep_feature.py
def drag_and_drop_button_example2():
	class MyButton(QPushButton):
		def __init__(self, title, parent):
			super().__init__(title, parent)

		def mouseMoveEvent(self, event: QMouseEvent):
			if event.buttons() != Qt.RightButton:
				return

			drag = QDrag(self)

			mime_data = QMimeData()
			mime_data.setData('application/hotspot', b'%d %d' % (event.x(), event.y()))
			drag.setMimeData(mime_data)

			pixmap = QPixmap(self.size())
			self.render(pixmap)
			drag.setPixmap(pixmap)

			drag.setHotSpot(event.pos() - self.rect().topLeft())

			drag.exec_(Qt.MoveAction)

	class MyForm(QWidget):
		def __init__(self, parent=None):
			super().__init__(parent, flags=Qt.Widget)

			self.setWindowTitle('Drag and Drop')
			self.setFixedSize(300, 300)
			self.setAcceptDrops(True)

			self.init_ui()

		def init_ui(self):
			self.btn = MyButton('Drag me to the moon', self)
			self.btn.show()

		def dragEnterEvent(self, event: QDragEnterEvent):
			event.accept()

		def dropEvent(self, event: QDropEvent):
			if event.mimeData().hasFormat('application/hotspot'):
				position = event.pos()

				offset = event.mimeData().data('application/hotspot')
				x, y = offset.data().decode('utf-8').split()
				self.btn.move(position - QPoint(int(x), int(y)))

				event.setDropAction(Qt.MoveAction)
				event.accept()
			else:
				event.ignore()

	#--------------------
	app = QApplication(sys.argv)

	form = MyForm()
	form.show()

	sys.exit(app.exec_())

def main():
	#drag_and_drop_text_example()
	#drag_and_drop_button_example1()
	drag_and_drop_button_example2()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
