#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
from PySide2.QtWidgets import QApplication

def hello_world_tutorial():
	from PySide2.QtWidgets import QLabel

	app = QApplication(sys.argv)
	#app = QApplication([])

	#label = QLabel('Hello World!')
	label = QLabel('<font color=red size=40>Hello World!</font>')
	label.show()

	app.exec_()

def qml_tutorial():
	from PySide2.QtCore import QUrl
	from PySide2.QtQuick import QQuickView

	app = QApplication([])

	url = QUrl('./view.qml')
	view = QQuickView()
	view.setSource(url)
	view.show()

	app.exec_()

def button_tutorial():
	from PySide2.QtCore import Slot
	from PySide2.QtWidgets import QPushButton

	@Slot()
	def say_hello():
		print('Button clicked, Hello!')

	app = QApplication(sys.argv)

	# Create a button, connect it and show it.
	button = QPushButton('Click me')
	button.clicked.connect(say_hello)
	button.show()

	# Run the main Qt loop.
	app.exec_()

def dialog_tutorial():
	from PySide2.QtWidgets import QDialog, QPushButton, QLineEdit, QVBoxLayout

	class MyForm(QDialog):
		def __init__(self, parent=None):
			super(MyForm, self).__init__(parent)
			self.setWindowTitle('My Form')

			# Create widgets.
			self.edit = QLineEdit('Write my name here')
			self.button = QPushButton('Show Greetings')

			# Create layout and add widgets.
			layout = QVBoxLayout()
			layout.addWidget(self.edit)
			layout.addWidget(self.button)
			# Set dialog layout.
			self.setLayout(layout)

			# Add button signal to greetings slot.
			self.button.clicked.connect(self.greet)

		# Greets the user.
		def greet(self):
			print ('Hello {}'.format(self.edit.text()))

	app = QApplication(sys.argv)

	# Create and show the form.
	form = MyForm()
	form.show()

	# Run the main Qt loop.
	sys.exit(app.exec_())

def ui_file_tutorial_1():
	from PySide2.QtWidgets import QMainWindow
	from ui_mainwindow import Ui_MainWindow

	class MyMainWindow(QMainWindow):
		def __init__(self):
			super(MyMainWindow, self).__init__()
			self.ui = Ui_MainWindow()
			self.ui.setupUi(self)

	app = QApplication(sys.argv)

	window = MyMainWindow()
	window.show()

	sys.exit(app.exec_())

def ui_file_tutorial_2():
	from PySide2.QtCore import QFile, QIODevice
	from PySide2.QtUiTools import QUiLoader

	app = QApplication(sys.argv)

	ui_file_name = './mainwindow.ui'

	ui_file = QFile(ui_file_name)
	if not ui_file.open(QIODevice.ReadOnly):
		print('Cannot open {}: {}'.format(ui_file_name, ui_file.errorString()))
		sys.exit(-1)

	loader = QUiLoader()
	window = loader.load(ui_file)
	ui_file.close()

	if not window:
		print(loader.errorString())
		sys.exit(-1)
	window.show()

	sys.exit(app.exec_())

def qrc_file_tutorial():
	from PySide2.QtGui import QIcon, QPixmap
	from PySide2.QtWidgets import QMainWindow, QToolBar
	import rc_icons

	class MyMainWindow(QMainWindow):
		def __init__(self):
			super(MyMainWindow, self).__init__()

			toolBar = QToolBar()
			self.addToolBar(toolBar)

			playMenu = self.menuBar().addMenu('&Play')

			playIcon = QIcon(QPixmap(':/icons/play.png'))
			self.playAction = toolBar.addAction(playIcon, 'Play')
			#self.playAction.triggered.connect(self.player.play)
			playMenu.addAction(self.playAction)

			previousIcon = QIcon(QPixmap(':/icons/backward.png'))
			self.previousAction = toolBar.addAction(previousIcon, 'Previous')
			#self.previousAction.triggered.connect(self.player.previous)
			playMenu.addAction(self.previousAction)

			pauseIcon = QIcon(QPixmap(':/icons/pause.png'))
			self.pauseAction = toolBar.addAction(pauseIcon, 'Pause')
			#self.pauseAction.triggered.connect(self.player.pause)
			playMenu.addAction(self.pauseAction)

			nextIcon = QIcon(QPixmap(':/icons/forward.png'))
			self.nextAction = toolBar.addAction(nextIcon, 'Next')
			#self.nextAction.triggered.connect(self.player.next)
			playMenu.addAction(self.nextAction)

			stopIcon = QIcon(QPixmap(':/icons/stop.png'))
			self.stopAction = toolBar.addAction(stopIcon, 'Stop')
			#self.stopAction.triggered.connect(self.player.stop)
			playMenu.addAction(self.stopAction)

	app = QApplication(sys.argv)

	window = MyMainWindow()
	window.show()

	sys.exit(app.exec_())

def widget_styling_tutorial_1():
	from PySide2.QtCore import Qt
	from PySide2.QtWidgets import QLabel

	app = QApplication()

	w = QLabel('This is a placeholder text')
	w.setAlignment(Qt.AlignCenter)
	if False:
		w.setStyleSheet("""
			background-color: #262626;
			color: #FFFFFF;
			font-family: Titillium;
			font-size: 18px;
		""")
	w.show()

	if True:
		with open('./style1.qss', 'r') as fd:
			style = fd.read()
			app.setStyleSheet(style)

	sys.exit(app.exec_())

def widget_styling_tutorial_2():
	from PySide2.QtCore import Qt
	from PySide2.QtWidgets import QWidget, QListWidget, QListWidgetItem, QLabel, QPushButton, QVBoxLayout, QHBoxLayout

	class MyWidget(QWidget):
		def __init__(self, parent=None):
			super(MyWidget, self).__init__(parent)

			menu_widget = QListWidget()
			for i in range(10):
				item = QListWidgetItem('Item {}'.format(i))
				item.setTextAlignment(Qt.AlignCenter)
				menu_widget.addItem(item)

			text = """
			This displays a two column widget,
			with a QListWidget on the left and a QLabel and a QPushButton on the right.
			It looks like this when you run the code:
			"""
			text_widget = QLabel(text)
			button = QPushButton('Something')

			content_layout = QVBoxLayout()
			content_layout.addWidget(text_widget)
			content_layout.addWidget(button)
			main_widget = QWidget()
			main_widget.setLayout(content_layout)

			layout = QHBoxLayout()
			layout.addWidget(menu_widget, 1)
			layout.addWidget(main_widget, 4)
			self.setLayout(layout)

	app = QApplication()

	w = MyWidget()
	w.show()

	if True:
		with open('./style2.qss', 'r') as fd:
			style = fd.read()
			app.setStyleSheet(style)

	sys.exit(app.exec_())

def main():
	# REF [site] >> https://doc.qt.io/qtforpython/tutorials/

	#hello_world_tutorial()
	#qml_tutorial()
	#button_tutorial()
	#dialog_tutorial()

	# Using QtCreator:
	#	pyside2-uic mainwindow.ui > ui_mainwindow.py
	#	pyside2-rcc icons.qrc -o rc_icons.py

	# Generate a Python class.
	#ui_file_tutorial_1()
	# Load directly.
	#ui_file_tutorial_2()

	# Generate a Python class.
	#qrc_file_tutorial()

	#widget_styling_tutorial_1()
	widget_styling_tutorial_2()

#--------------------------------------------------------------------

# Usage:
#	python qt_basic.py
#	QT_XCB_GL_INTEGRATION=none python qt_basic.py

if '__main__' == __name__:
	main()
