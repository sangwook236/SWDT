#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
from PyQt5.QtWidgets import QApplication

def hello_world_tutorial():
	from PyQt5.QtWidgets import QLabel

	app = QApplication(sys.argv)
	#app = QApplication([])

	#label = QLabel('Hello World!')
	label = QLabel('<font color=red size=40>Hello World!</font>')
	label.show()

	sys.exit(app.exec_())

def qml_tutorial():
	from PyQt5.QtCore import QUrl
	from PyQt5.QtQuick import QQuickView

	app = QApplication([])

	url = QUrl('./view.qml')
	view = QQuickView()
	view.setSource(url)
	view.show()

	sys.exit(app.exec_())

def button_tutorial():
	from PyQt5.QtCore import Slot
	from PyQt5.QtWidgets import QPushButton

	@Slot()
	def say_hello():
		print('Button clicked, Hello!')

	app = QApplication(sys.argv)

	# Create a button, connect it and show it.
	button = QPushButton('Click me')
	button.clicked.connect(say_hello)
	button.show()

	# Run the main Qt loop.
	sys.exit(app.exec_())

def dialog_tutorial():
	from PyQt5.QtWidgets import QDialog, QPushButton, QLineEdit, QVBoxLayout

	class MyForm(QDialog):
		def __init__(self, parent=None):
			super().__init__(parent)
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

	#--------------------
	app = QApplication(sys.argv)

	# Create and show the form.
	form = MyForm()
	form.show()

	# Run the main Qt loop.
	sys.exit(app.exec_())

# REF [site] >> https://realpython.com/python-menus-toolbars/
def menu_example():
	import functools
	from PyQt5.QtCore import Qt
	from PyQt5.QtGui import QIcon, QKeySequence
	from PyQt5.QtWidgets import QMainWindow, QLabel, QSpinBox, QMenuBar, QMenu, QToolBar, QAction

	class MyMainWindow(QMainWindow):
		def __init__(self, parent=None):
			super().__init__(parent=parent)

			self._initUi()
			#self._createContextMenu()

		def _initUi(self):
			self.setWindowTitle('Menus & Toolbars')
			self.setWindowIcon(QIcon('pyicon.png'))

			# Window dimensions.
			self.setGeometry(100, 100, 600, 400)
			#self.resize(800, 800)
			#geometry = QApplication.instance().desktop().availableGeometry(self)
			#self.setFixedSize(geometry.width() * 0.8, geometry.height() * 0.7)
			#self.setGeometry(geometry.width() * 0.05, geometry.height() * 0.05, geometry.width() * 0.9, geometry.height() * 0.9)

			label = QLabel('Hello, World')
			label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
			self.setCentralWidget(label)

			#--------------------
			# Action.
			self.newAction = QAction(self)
			self.newAction.setText('&New')
			self.newAction.setIcon(QIcon(':file-new.svg'))
			self.openAction = QAction(QIcon(':file-open.svg'), '&Open...', self)
			self.saveAction = QAction(QIcon(':file-save.svg'), '&Save', self)
			self.exitAction = QAction('&Exit', self)

			newTip = 'Create a new file'
			self.newAction.setStatusTip(newTip)
			self.newAction.setToolTip(newTip)

			self.cutAction = QAction(QIcon(':edit-cut.svg'), 'C&ut', self)
			self.copyAction = QAction(QIcon(':edit-copy.svg'), '&Copy', self)
			self.pasteAction = QAction(QIcon(':edit-paste.svg'), '&Paste', self)

			self.helpContentAction = QAction('&Help Content', self)
			self.aboutAction = QAction('&About', self)

			self.newAction.setShortcut('Ctrl+N')
			self.openAction.setShortcut('Ctrl+O')
			self.saveAction.setShortcut('Ctrl+S')
			self.exitAction.setShortcut(QKeySequence.Quit)

			self.cutAction.setShortcut(QKeySequence.Cut)
			self.copyAction.setShortcut(QKeySequence.Copy)
			self.pasteAction.setShortcut(QKeySequence.Paste)

			self.newAction.triggered.connect(self.newFile)
			self.openAction.triggered.connect(self.openFile)
			self.saveAction.triggered.connect(self.saveFile)
			self.exitAction.triggered.connect(self.close)

			self.copyAction.triggered.connect(self.copyContent)
			self.pasteAction.triggered.connect(self.pasteContent)
			self.cutAction.triggered.connect(self.cutContent)

			self.helpContentAction.triggered.connect(self.helpContent)
			self.aboutAction.triggered.connect(self.about)

			#--------------------
			# Menu bar.
			#menuBar = QMenuBar(self)
			#self.setMenuBar(menuBar)
			menuBar = self.menuBar()

			fileMenu = QMenu('&File', self)
			menuBar.addMenu(fileMenu)
			fileMenu.addAction(self.newAction)
			fileMenu.addAction(self.openAction)
			self.openRecentMenu = fileMenu.addMenu('Open Recent')
			fileMenu.addAction(self.saveAction)
			fileMenu.addSeparator()
			fileMenu.addAction(self.exitAction)

			self.openRecentMenu.aboutToShow.connect(self.populateOpenRecent)

			editMenu = menuBar.addMenu('&Edit')
			editMenu.addAction(self.cutAction)
			editMenu.addAction(self.copyAction)
			editMenu.addAction(self.pasteAction)
			editMenu.addSeparator()
			findMenu = editMenu.addMenu('Find and Replace')
			findMenu.addAction('Find...')
			findMenu.addAction('Replace...')

			helpMenu = menuBar.addMenu(QIcon(':help-content.svg'), '&Help')
			helpMenu.addAction(self.helpContentAction)
			helpMenu.addAction(self.aboutAction)

			#--------------------
			# Toolbar.
			fileToolBar = self.addToolBar('File')
			fileToolBar.setMovable(False)
			fileToolBar.addAction(self.newAction)
			fileToolBar.addAction(self.openAction)
			fileToolBar.addAction(self.saveAction)

			editToolBar = QToolBar('Edit', self)
			self.addToolBar(editToolBar)
			editToolBar.addAction(self.copyAction)
			editToolBar.addAction(self.pasteAction)
			editToolBar.addAction(self.cutAction)
			self.fontSizeSpinBox = QSpinBox()
			self.fontSizeSpinBox.setFocusPolicy(Qt.NoFocus)
			editToolBar.addWidget(self.fontSizeSpinBox)

			helpToolBar = QToolBar('Help', self)
			self.addToolBar(Qt.LeftToolBarArea, helpToolBar)

			#--------------------
			# Status bar.
			#statusBar = QStatusBar()
			#self.setStatusBar(statusBar)
			statusBar = self.statusBar()

			# Add a temporary message.
			statusBar.showMessage('Ready', msecs=3000)
			# Add a permanent message.
			wcLabel = QLabel(f'{self.getWordCount()} Words')
			statusBar.addPermanentWidget(wcLabel)

		def contextMenuEvent(self, event):
			menu = QMenu(self.centralWidget())

			menu.addAction(self.newAction)
			menu.addAction(self.openAction)
			menu.addAction(self.saveAction)
			separator = QAction(self)
			separator.setSeparator(True)
			menu.addAction(separator)
			menu.addAction(self.copyAction)
			menu.addAction(self.pasteAction)
			menu.addAction(self.cutAction)

			menu.exec(event.globalPos())

		def _createContextMenu(self):
			self.centralWidget().setContextMenuPolicy(Qt.ActionsContextMenu)

			self.centralWidget().addAction(self.newAction)
			self.centralWidget().addAction(self.openAction)
			self.centralWidget().addAction(self.saveAction)
			separator = QAction(self)
			separator.setSeparator(True)
			self.centralWidget().addAction(separator)
			self.centralWidget().addAction(self.cutAction)
			self.centralWidget().addAction(self.copyAction)
			self.centralWidget().addAction(self.pasteAction)

		def newFile(self):
			self.centralWidget().setText('<b>File > New</b> clicked')

		def openFile(self):
			self.centralWidget().setText('<b>File > Open...</b> clicked')

		def saveFile(self):
			self.centralWidget().setText('<b>File > Save</b> clicked')

		def copyContent(self):
			self.centralWidget().setText('<b>Edit > Copy</b> clicked')

		def pasteContent(self):
			self.centralWidget().setText('<b>Edit > Paste</b> clicked')

		def cutContent(self):
			self.centralWidget().setText('<b>Edit > Cut</b> clicked')

		def helpContent(self):
			self.centralWidget().setText('<b>Help > Help Content...</b> clicked')

		def about(self):
			self.centralWidget().setText('<b>Help > About...</b> clicked')

		def openRecentFile(self, filename):
			self.centralWidget().setText(f'<b>{filename}</b> opened')

		def populateOpenRecent(self):
			# Step 1. Remove the old options from the menu.
			self.openRecentMenu.clear()
			# Step 2. Dynamically create the actions.
			actions = []
			filenames = [f'File-{n}' for n in range(5)]
			for filename in filenames:
				action = QAction(filename, self)
				action.triggered.connect(functools.partial(self.openRecentFile, filename))
				actions.append(action)
			# Step 3. Add the actions to the menu.
			self.openRecentMenu.addActions(actions)

		def getWordCount(self):
			return 42

	#--------------------
	app = QApplication(sys.argv)

	window = MyMainWindow()
	window.show()

	sys.exit(app.exec_())

# REF [site] >> https://pythonspot.com/pyqt5-file-dialog/
def common_dialog_example():
	from PyQt5.QtWidgets import QWidget, QFileDialog

	class MyWiget(QWidget):
		def __init__(self, parent=None):
			super().__init__(parent=parent)

			self._initUi()

		def _initUi(self):
			self.setWindowTitle('Common Dialog')
			self.setGeometry(100, 100, 800, 600)
			
			self.openFileNameDialog()
			self.openFileNamesDialog()
			self.saveFileDialog()
			self.openDirectoryDialog()

			self.show()

		def openFileNameDialog(self):
			options = QFileDialog.Options()
			options |= QFileDialog.DontUseNativeDialog
			filepath, _ = QFileDialog.getOpenFileName(self, caption='Open a File', directory='', filter='Python Files (*.py);;All Files (*)', options=options)
			#filepath, _ = QFileDialog.getOpenFileUrl(self, caption='Open a URL', directory='', filter='Python Files (*.py);;All Files (*)', options=options)
			if filepath:
				print(filepath)

		def openFileNamesDialog(self):
			options = QFileDialog.Options()
			options |= QFileDialog.DontUseNativeDialog
			filepaths, _ = QFileDialog.getOpenFileNames(self, caption='Open Files', directory='', filter='Python Files (*.py);;All Files (*)', options=options)
			#filepaths, _ = QFileDialog.getOpenFileNames(self, caption='Open URLs', directory='', filter='Python Files (*.py);;All Files (*)', options=options)
			if filepaths:
				print(filepaths)

		def saveFileDialog(self):
			options = QFileDialog.Options()
			options |= QFileDialog.DontUseNativeDialog
			filepath, _ = QFileDialog.getSaveFileName(self, caption='Save a File', directory='', filter='Text Files (*.txt);;All Files (*)', options=options)
			#filepath, _ = QFileDialog.getSaveFileUrl(self, caption='Save a URL', directory='', filter='Text Files (*.txt);;All Files (*)', options=options)
			if filepath:
				print(filepath)

		def openDirectoryDialog(self):
			options = QFileDialog.Options()
			options |= QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog
			dir = QFileDialog.getExistingDirectory(self, caption='Open a Directory', directory='', options=options)
			#dir = QFileDialog.getExistingDirectoryUrl(self, caption='Open a Directory URL', options=options)
			if dir:
				print(dir)

	#--------------------
	app = QApplication(sys.argv)

	widget = MyWiget()
	widget.show()

	sys.exit(app.exec_())

def custom_dialog_example():
	from PyQt5.QtCore import Qt
	from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QLabel, QLineEdit, QCheckBox, QRadioButton, QComboBox, QHBoxLayout, QVBoxLayout

	class MyDialog(QDialog):
		def __init__(self, parent=None):
			super().__init__(parent)

			self._initUi()

		def _initUi(self):
			self.setWindowTitle('Custom Dialog')
			self.resize(300, 200)

			self.labelName = QLabel('Name:')
			self.editName = QLineEdit('My Name')
			self.checkKorean = QCheckBox('Korean')
			self.radioMale = QRadioButton('Male')
			self.radioMale.setChecked(True)
			self.radioFemale = QRadioButton('Female')
			self.labelLanguage = QLabel('Language:')
			self.comboLanguage = QComboBox()
			self.comboLanguage.addItem('Korean')
			self.comboLanguage.addItem('Englis')
			self.comboLanguage.addItems(['Chinese', 'Japanese'])
			self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)

			vLayout = QVBoxLayout()
			#vLayout.addStretch(1)

			hLayout = QHBoxLayout()
			#hLayout.addStretch(1)
			hLayout.addWidget(self.labelName)
			hLayout.addWidget(self.editName)
			vLayout.addLayout(hLayout)
			vLayout.addWidget(self.checkKorean)
			hLayout = QHBoxLayout()
			hLayout.addWidget(self.radioMale)
			hLayout.addWidget(self.radioFemale)
			vLayout.addLayout(hLayout)
			hLayout = QHBoxLayout()
			#hLayout.setAlignment(Qt.AlignJustify)
			hLayout.addWidget(self.labelLanguage)
			hLayout.addWidget(self.comboLanguage)
			vLayout.addLayout(hLayout)
			vLayout.addWidget(self.buttonBox)

			self.setLayout(vLayout)

			self.checkKorean.toggled.connect(lambda: self.onKorenCheckBox())
			self.radioMale.toggled.connect(lambda: self.onGenderRadioButton(self.radioMale))
			self.radioFemale.toggled.connect(lambda: self.onGenderRadioButton(self.radioFemale))
			self.comboLanguage.currentIndexChanged.connect(self.onLanguageComboBox)
			self.buttonBox.button(QDialogButtonBox.Ok).clicked.connect(self.accept)
			self.buttonBox.button(QDialogButtonBox.Cancel).clicked.connect(self.reject)

		def onKorenCheckBox(self):
			if self.checkKorean.isChecked():
				print('Korean is checked.')
			else:
				print('Korean is unchecked.')

		def onGenderRadioButton(self, radioButton):
			if radioButton == self.radioMale:
				if radioButton.isChecked() == True:
					print('Male is selected.')
			elif radioButton == self.radioFemale:
				if radioButton.isChecked() == True:
					print('Female is selected.')

		def onLanguageComboBox(self, index):
			print('The {}-th item, {} is selected.'.format(index, self.comboLanguage.currentText()))

		def accept(self):
			print('Name: {}.'.format(self.editName.text()))
			print('Accept.')
			super().accept()

		def reject(self):
			print('Reject.')
			super().reject()

	#--------------------
	app = QApplication(sys.argv)

	dlg = MyDialog()
	dlg.show()

	sys.exit(app.exec_())

def ui_file_tutorial_1():
	from PySide2.QtWidgets import QMainWindow
	from ui_mainwindow import Ui_MainWindow

	class MyMainWindow(QMainWindow):
		def __init__(self):
			super().__init__()
			self.ui = Ui_MainWindow()
			self.ui.setupUi(self)

	#--------------------
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

	if window:
		window.show()

		sys.exit(app.exec_())
	else:
		print(loader.errorString())
		sys.exit(-1)

def qrc_file_tutorial():
	from PySide2.QtGui import QIcon, QPixmap
	from PySide2.QtWidgets import QMainWindow, QToolBar
	import rc_icons

	class MyMainWindow(QMainWindow):
		def __init__(self):
			super().__init__()

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

	#--------------------
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
	from PyQt5.QtCore import Qt
	from PyQt5.QtWidgets import QWidget, QListWidget, QListWidgetItem, QLabel, QPushButton, QVBoxLayout, QHBoxLayout

	class MyWidget(QWidget):
		def __init__(self, parent=None):
			super().__init__(parent)

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

	#--------------------
	app = QApplication()

	w = MyWidget()
	w.show()

	if True:
		with open('./style2.qss', 'r') as fd:
			style = fd.read()
			app.setStyleSheet(style)

	sys.exit(app.exec_())

def sdi_example():
	from PyQt5.QtCore import Qt
	from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QTextEdit, QVBoxLayout, QAction
	from PyQt5.QtGui import QIcon, QImage, QPixmap, QKeySequence

	class MyWidget(QWidget):
		def __init__(self, parent=None):
			super().__init__(parent)

			if True:
				QTextEdit(self)
			elif False:
				layout = QVBoxLayout()
				layout.addWidget(QTextEdit())
				self.setLayout(layout)
			else:
				label = QLabel(self)

				image_filepath = './images/cat.jpg'
				if True:
					pixmap = QPixmap(image_filepath)
				else:
					image = QImage(image_filepath)
					pixmap = QPixmap(image)
				label.setPixmap(pixmap)
				#label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
				#label.setScaledContents(True)

				#layout = QVBoxLayout()
				#layout.addWidget(label)
				#self.setLayout(layout)

	class MyMainWindow(QMainWindow):
		def __init__(self):
			super().__init__()
			self.setWindowTitle('SDI Application')
			self.setWindowIcon(QIcon('pyicon.png'))

			# Window dimensions.
			self.setGeometry(100, 100, 800, 800)
			#self.resize(800, 800)
			#geometry = QApplication.instance().desktop().availableGeometry(self)
			#self.setFixedSize(geometry.width() * 0.8, geometry.height() * 0.7)

			#widget = QTextEdit()
			widget = MyWidget()
			self.setCentralWidget(widget)

			# Menu bar.
			menu_bar = self.menuBar()

			file_menu = menu_bar.addMenu('File')

			# Exit QAction.
			exit_action = QAction('Exit', self)
			exit_action.setShortcut(QKeySequence.Quit)
			exit_action.triggered.connect(self.close)

			file_menu.addAction(exit_action)

			# Status bar.
			status = self.statusBar()
			status.showMessage('Status bar')

	#--------------------
	app = QApplication(sys.argv)

	window = MyMainWindow()
	window.show()

	sys.exit(app.exec_())

# REF [site] >> https://codeloop.org/python-multi-document-interface-with-pyside2/
def mdi_example():
	from PyQt5.QtWidgets import QMainWindow, QWidget, QMdiArea, QMdiSubWindow, QTextEdit, QVBoxLayout, QAction
	from PyQt5.QtGui import QIcon, QKeySequence

	class MyWidget(QWidget):
		def __init__(self, parent=None):
			super().__init__(parent)

			layout = QVBoxLayout()
			layout.addWidget(QTextEdit())
			self.setLayout(layout)

	class MyMainWindow(QMainWindow):
		count = 0

		def __init__(self, widgets):
			super().__init__()
			self.widget_classes = widget_classes

			self.init_ui()

		def init_ui(self):
			self.setWindowTitle('MDI Application')
			self.setWindowIcon(QIcon('pyicon.png'))
			self.setGeometry(100, 100, 800, 600)

			# Create an instance of MDI.
			self.mdi = QMdiArea()
			self.setCentralWidget(self.mdi)

			# Menu bar.
			menu_bar = self.menuBar()

			file_menu = menu_bar.addMenu('File')
			file_menu.addAction('New')
			file_menu.addAction('Cascade')
			file_menu.addAction('Tiled')

			file_menu.triggered[QAction].connect(self.window_triggered)

			# Exit QAction.
			exit_action = QAction('Exit', self)
			exit_action.setShortcut(QKeySequence.Quit)
			exit_action.triggered.connect(self.close)

			file_menu.addAction(exit_action)

			# Status bar.
			status = self.statusBar()
			status.showMessage('Status bar')

		def window_triggered(self, p):
			if p.text() == 'New':
				MyMainWindow.count += 1

				sub = QMdiSubWindow()
				widget = self.widget_classes[MyMainWindow.count % 2]()
				sub.setWidget(widget)
				sub.setWindowTitle('Sub Window {}'.format(MyMainWindow.count))

				self.mdi.addSubWindow(sub)
				sub.show()

			if p.text() == 'Cascade':
				self.mdi.cascadeSubWindows()

			if p.text() == 'Tiled':
				self.mdi.tileSubWindows()

	#--------------------
	app = QApplication(sys.argv)

	widget_classes = [QTextEdit, MyWidget]
	window = MyMainWindow(widget_classes)
	window.show()

	sys.exit(app.exec_())

def main():
	# REF [site] >> https://doc.qt.io/qtforpython/tutorials/

	#hello_world_tutorial()
	#qml_tutorial()
	#button_tutorial()
	#dialog_tutorial()

	#menu_example()
	#common_dialog_example()
	custom_dialog_example()

	# Using QtCreator:
	#	PySide2:
	#		pyside2-uic mainwindow.ui > ui_mainwindow.py
	#		pyside2-rcc icons.qrc -o rc_icons.py
	#	PyQt5:
	#		pyuic5 mainwindow.ui > ui_mainwindow.py
	#		pyrcc5 icons.qrc -o rc_icons.py

	# Generate a Python class.
	#ui_file_tutorial_1()
	# Load directly.
	#ui_file_tutorial_2()

	# Generate a Python class.
	#qrc_file_tutorial()

	#widget_styling_tutorial_1()
	#widget_styling_tutorial_2()

	#--------------------
	#sdi_example()
	#mdi_example()

#--------------------------------------------------------------------

# Usage:
#	python qt_basic.py
#	QT_XCB_GL_INTEGRATION=none python qt_basic.py

if '__main__' == __name__:
	main()
