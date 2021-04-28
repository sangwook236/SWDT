#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://stackoverflow.com/questions/59239074/how-to-translate-drag-move-qgraphicsscene
def graphics_view_example():
	from PyQt5.QtWidgets import QApplication, QFrame, QDialog, QVBoxLayout, QSizePolicy
	from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsPolygonItem
	from PyQt5.QtGui import QPolygonF, QColor, QPen, QBrush, QFont
	from PyQt5.QtCore import Qt, QPointF, pyqtSignal

	class MyGraphicsButtonItem(QGraphicsPolygonItem):
		def __init__(self, parent=None):
			super().__init__(parent=parent)

			points = [
				[60.1, 19.6, 0.0], [60.1, 6.5, 0.0], [60.1, -6.5, 0.0], [60.1, -19.6, 0.0], [60.1, -19.6, 0.0],
				[20.0, -19.6, 0.0], [-20, -19.6, 0.0], [-60.1, -19.6, 0.0], [-60.1, -19.6, 0.0], [-60.1, -6.5, 0.0],
				[-60.1, 6.5, 0.0], [-60.1, 19.6, 0.0], [-60.1, 19.6, 0.0], [-20.0, 19.6, 0.0], [20.0, 19.6, 0.0],
				[60.1, 19.6, 0.0]
			]
			polygon = QPolygonF([QPointF(v1, v2) for v1, v2, v3 in points])

			self.setPolygon(polygon)
			self.setPen(QPen(QColor(0, 0, 0), 0, Qt.SolidLine, Qt.FlatCap, Qt.MiterJoin))
			self.setBrush(QColor(220, 40, 30))

	class MyGraphicsView(QGraphicsView):
		zoom_signal = pyqtSignal(bool)

		def __init__(self, parent=None):
			super().__init__(parent)

			self._scene = QGraphicsScene(backgroundBrush=Qt.gray)
			self._zoom = 0

			self.setScene(self._scene)
			self.setTransformationAnchor(QGraphicsView.NoAnchor)
			#self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
			self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
			self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
			self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
			self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
			self.setFrameShape(QFrame.NoFrame)
			self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

			button = MyGraphicsButtonItem()
			self._scene.addItem(button)

			text = self._scene.addText('Hello, World!', QFont('Arial', 20))
			text.setDefaultTextColor(Qt.white)
			text.setFlag(QGraphicsItem.ItemIsMovable)  # Movable text.

			self.startPos = None

		def mousePressEvent(self, event):
			if event.modifiers() & Qt.ControlModifier and event.button() == Qt.LeftButton:
				self.startPos = event.pos()
			else:
				super().mousePressEvent(event)

		def mouseMoveEvent(self, event):
			if self.startPos is not None:
				delta = self.startPos - event.pos()

				# Get the current transformation (which is a matrix that includes the scaling ratios m11 refers to the horizontal scale, m22 to the vertical scale).
				transform = self.transform()

				# Divide the delta by their corresponding ratio.
				deltaX = delta.x() / transform.m11()
				deltaY = delta.y() / transform.m22()

				# Translate the current sceneRect by the delta.
				self.setSceneRect(self.sceneRect().translated(deltaX, deltaY))

				self.startPos = event.pos()
			else:
				super().mouseMoveEvent(event)

		def mouseReleaseEvent(self, event):
			self.startPos = None
			super().mouseReleaseEvent(event)

	class MyDialog(QDialog):
		def __init__(self, parent=None):
			super().__init__(parent)

			self.init_ui()

		def init_ui(self):
			layout = QVBoxLayout()
			graphics = MyGraphicsView()
			layout.addWidget(graphics)
			self.setLayout(layout)

	#--------------------
	import sys

	app = QApplication(sys.argv)

	dlg = MyDialog()
	dlg.setGeometry(500, 100, 500, 900)
	dlg.show()

	sys.exit(app.exec_())

# REF [site] >> https://stackoverflow.com/questions/35508711/how-to-enable-pan-and-zoom-in-a-qgraphicsview
def simple_photo_viewer_example():
	from PyQt5 import QtCore, QtGui, QtWidgets

	class MySimplePhotoViewer(QtWidgets.QGraphicsView):
		photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

		def __init__(self, parent=None):
			super().__init__(parent)

			self._zoom = 0
			self._empty = True
			self._scene = QtWidgets.QGraphicsScene(self)
			self._photo = QtWidgets.QGraphicsPixmapItem()
			self._scene.addItem(self._photo)

			self.setScene(self._scene)
			self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
			self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
			self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
			self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
			self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
			self.setFrameShape(QtWidgets.QFrame.NoFrame)

		def hasPhoto(self):
			return not self._empty

		def fitInView(self, scale=True):
			rect = QtCore.QRectF(self._photo.pixmap().rect())
			if not rect.isNull():
				self.setSceneRect(rect)
				if self.hasPhoto():
					unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
					self.scale(1 / unity.width(), 1 / unity.height())
					viewrect = self.viewport().rect()
					scenerect = self.transform().mapRect(rect)
					factor = min(viewrect.width() / scenerect.width(), viewrect.height() / scenerect.height())
					self.scale(factor, factor)
				self._zoom = 0

		def setPhoto(self, pixmap=None):
			self._zoom = 0
			if pixmap and not pixmap.isNull():
				self._empty = False
				self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
				self._photo.setPixmap(pixmap)
			else:
				self._empty = True
				self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
				self._photo.setPixmap(QtGui.QPixmap())
			self.fitInView()

		def wheelEvent(self, event):
			if self.hasPhoto():
				if event.angleDelta().y() > 0:
					factor = 1.25
					self._zoom += 1
				else:
					factor = 0.8
					self._zoom -= 1
				if self._zoom > 0:
					self.scale(factor, factor)
				elif self._zoom == 0:
					self.fitInView()
				else:
					self._zoom = 0

		def toggleDragMode(self):
			if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
				self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
			elif not self._photo.pixmap().isNull():
				self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

		def mousePressEvent(self, event):
			if self._photo.isUnderMouse():
				self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
			super().mousePressEvent(event)

	class MyWidget(QtWidgets.QWidget):
		def __init__(self, parent=None):
			super().__init__(parent)

			self.viewer = MySimplePhotoViewer(self)

			# Load image.
			self.btnLoad = QtWidgets.QToolButton(self)
			self.btnLoad.setText('Load image')
			self.btnLoad.clicked.connect(self.loadImage)

			# Toggle between dragging/panning and getting pixel info.
			self.btnPixInfo = QtWidgets.QToolButton(self)
			self.btnPixInfo.setText('Toggle manipulation mode')
			self.btnPixInfo.clicked.connect(self.pixInfo)
			self.editPixInfo = QtWidgets.QLineEdit(self)
			self.editPixInfo.setReadOnly(True)
			self.viewer.photoClicked.connect(self.photoClicked)

			# Arrange layout.
			vb_layout = QtWidgets.QVBoxLayout(self)
			vb_layout.addWidget(self.viewer)
			hb_layout = QtWidgets.QHBoxLayout()
			hb_layout.setAlignment(QtCore.Qt.AlignLeft)
			hb_layout.addWidget(self.btnLoad)
			hb_layout.addWidget(self.btnPixInfo)
			hb_layout.addWidget(self.editPixInfo)
			hb_layout.addLayout(hb_layout)

		def loadImage(self):
			self.viewer.setPhoto(QtGui.QPixmap('./page.png'))

		def pixInfo(self):
			self.viewer.toggleDragMode()

		def photoClicked(self, pos):
			if self.viewer.dragMode() == QtWidgets.QGraphicsView.NoDrag:
				self.editPixInfo.setText('{:d}, {:d}'.format(pos.x(), pos.y()))

	#--------------------
	import sys

	app = QtWidgets.QApplication(sys.argv)

	widget = MyWidget()
	widget.setGeometry(500, 300, 800, 600)
	widget.show()

	sys.exit(app.exec_())

# REF [site] >> https://github.com/marcel-goldschen-ohm/PyQtImageViewer
def simple_image_viewer_example():
	import os
	from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget, QVBoxLayout
	from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene
	from PyQt5.QtGui import QImage, QPixmap, QPainterPath
	from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QT_VERSION_STR

	class QtImageViewer(QGraphicsView):
		""" PyQt image viewer widget for a QPixmap in a QGraphicsView scene with mouse zooming and panning.

		Displays a QImage or QPixmap (QImage is internally converted to a QPixmap).
		To display any other image format, you must first convert it to a QImage or QPixmap.

		Some useful image format conversion utilities:
			qimage2ndarray: NumPy ndarray <==> QImage    (https://github.com/hmeine/qimage2ndarray)
			ImageQt: PIL Image <==> QImage  (https://github.com/python-pillow/Pillow/blob/master/PIL/ImageQt.py)

		Mouse interaction:
			Left mouse button drag: Pan image.
			Right mouse button drag: Zoom box.
			Right mouse button doubleclick: Zoom to show entire image.
		"""

		# Mouse button signals emit image scene (x, y) coordinates.
		# !!! For image (row, column) matrix indexing, row = y and column = x.
		leftMouseButtonPressed = pyqtSignal(float, float)
		rightMouseButtonPressed = pyqtSignal(float, float)
		leftMouseButtonReleased = pyqtSignal(float, float)
		rightMouseButtonReleased = pyqtSignal(float, float)
		leftMouseButtonDoubleClicked = pyqtSignal(float, float)
		rightMouseButtonDoubleClicked = pyqtSignal(float, float)

		def __init__(self, parent=None):
			super().__init__(parent)

			# Image is displayed as a QPixmap in a QGraphicsScene attached to this QGraphicsView.
			self.scene = QGraphicsScene(self)
			self.setScene(self.scene)

			self.zoom = 1

			# Store a local handle to the scene's current image pixmap.
			self._pixmapHandle = None

			# Image aspect ratio mode.
			# !!! ONLY applies to full image. Aspect ratio is always ignored when zooming.
			#   Qt.IgnoreAspectRatio: Scale image to fit viewport.
			#   Qt.KeepAspectRatio: Scale image to fit inside viewport, preserving aspect ratio.
			#   Qt.KeepAspectRatioByExpanding: Scale image to fill the viewport, preserving aspect ratio.
			self.aspectRatioMode = Qt.KeepAspectRatio

			# Scroll bar behaviour.
			#   Qt.ScrollBarAlwaysOff: Never shows a scroll bar.
			#   Qt.ScrollBarAlwaysOn: Always shows a scroll bar.
			#   Qt.ScrollBarAsNeeded: Shows a scroll bar only when zoomed.
			self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
			self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

			# Stack of QRectF zoom boxes in scene coordinates.
			self.zoomStack = []

			# Flags for enabling/disabling mouse interaction.
			self.canZoom = True
			self.canPan = True

		def hasImage(self):
			""" Returns whether or not the scene contains an image pixmap.
			"""
			return self._pixmapHandle is not None

		def clearImage(self):
			""" Removes the current image pixmap from the scene if it exists.
			"""
			if self.hasImage():
				self.scene.removeItem(self._pixmapHandle)
				self._pixmapHandle = None

		def pixmap(self):
			""" Returns the scene's current image pixmap as a QPixmap, or else None if no image exists.
			:rtype: QPixmap | None
			"""
			if self.hasImage():
				return self._pixmapHandle.pixmap()
			return None

		def image(self):
			""" Returns the scene's current image pixmap as a QImage, or else None if no image exists.
			:rtype: QImage | None
			"""
			if self.hasImage():
				return self._pixmapHandle.pixmap().toImage()
			return None

		def setImage(self, image):
			""" Set the scene's current image pixmap to the input QImage or QPixmap.
			Raises a RuntimeError if the input image has type other than QImage or QPixmap.
			:type image: QImage | QPixmap
			"""
			if type(image) is QPixmap:
				pixmap = image
			elif type(image) is QImage:
				pixmap = QPixmap.fromImage(image)
			else:
				raise RuntimeError('ImageViewer.setImage: Argument must be a QImage or QPixmap.')
			if self.hasImage():
				self._pixmapHandle.setPixmap(pixmap)
			else:
				self._pixmapHandle = self.scene.addPixmap(pixmap)
			self.setSceneRect(QRectF(pixmap.rect()))  # Set scene size to image size.
			self.updateViewer()

		def loadImageFromFile(self, fileName=''):
			""" Load an image from file.
			Without any arguments, loadImageFromFile() will popup a file dialog to choose the image file.
			With a fileName argument, loadImageFromFile(fileName) will attempt to load the specified image file directly.
			"""
			if len(fileName) == 0:
				if QT_VERSION_STR[0] == '4':
					fileName = QFileDialog.getOpenFileName(self, "Open image file.")
				elif QT_VERSION_STR[0] == '5':
					fileName, dummy = QFileDialog.getOpenFileName(self, "Open image file.")
			if len(fileName) and os.path.isfile(fileName):
				image = QImage(fileName)
				self.setImage(image)

		def updateViewer(self):
			""" Show current zoom (if showing entire image, apply current aspect ratio mode).
			"""
			if not self.hasImage():
				return
			if len(self.zoomStack) and self.sceneRect().contains(self.zoomStack[-1]):
				self.fitInView(self.zoomStack[-1], Qt.IgnoreAspectRatio)  # Show zoomed rect (ignore aspect ratio).
			else:
				self.zoomStack = []  # Clear the zoom stack (in case we got here because of an invalid zoom).
				self.fitInView(self.sceneRect(), self.aspectRatioMode)  # Show entire image (use current aspect ratio mode).

		def resizeEvent(self, event):
			""" Maintain current zoom on resize.
			"""
			self.updateViewer()

		def mousePressEvent(self, event):
			""" Start mouse pan or zoom mode.
			"""
			scenePos = self.mapToScene(event.pos())
			if event.button() == Qt.LeftButton:
				if self.canPan:
					self.setDragMode(QGraphicsView.ScrollHandDrag)
				self.leftMouseButtonPressed.emit(scenePos.x(), scenePos.y())
			elif event.button() == Qt.RightButton:
				if self.canZoom:
					self.setDragMode(QGraphicsView.RubberBandDrag)
				self.rightMouseButtonPressed.emit(scenePos.x(), scenePos.y())
			super().mousePressEvent(event)

		def mouseReleaseEvent(self, event):
			""" Stop mouse pan or zoom mode (apply zoom if valid).
			"""
			super().mouseReleaseEvent(event)
			scenePos = self.mapToScene(event.pos())
			if event.button() == Qt.LeftButton:
				self.setDragMode(QGraphicsView.NoDrag)
				self.leftMouseButtonReleased.emit(scenePos.x(), scenePos.y())
			elif event.button() == Qt.RightButton:
				if self.canZoom:
					viewBBox = self.zoomStack[-1] if len(self.zoomStack) else self.sceneRect()
					selectionBBox = self.scene.selectionArea().boundingRect().intersected(viewBBox)
					self.scene.setSelectionArea(QPainterPath())  # Clear current selection area.
					if selectionBBox.isValid() and (selectionBBox != viewBBox):
						self.zoomStack.append(selectionBBox)
						self.updateViewer()
				self.setDragMode(QGraphicsView.NoDrag)
				self.rightMouseButtonReleased.emit(scenePos.x(), scenePos.y())

		def mouseDoubleClickEvent(self, event):
			""" Show entire image.
			"""
			scenePos = self.mapToScene(event.pos())
			if event.button() == Qt.LeftButton:
				self.leftMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
			elif event.button() == Qt.RightButton:
				if self.canZoom:
					self.zoomStack = []  # Clear zoom stack.
					self.updateViewer()
				self.rightMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
			super().mouseDoubleClickEvent(event)

	class MyWidget(QWidget):
		def __init__(self, parent=None):
			super().__init__(parent)

			# Create an image viewer widget.
			self.view = QtImageViewer(self)
				
			# Set viewer's aspect ratio mode.
			# !!! ONLY applies to full image view.
			# !!! Aspect ratio always ignored when zoomed.
			#   Qt.IgnoreAspectRatio: Fit to viewport.
			#   Qt.KeepAspectRatio: Fit in viewport using aspect ratio.
			#   Qt.KeepAspectRatioByExpanding: Fill viewport using aspect ratio.
			self.view.aspectRatioMode = Qt.KeepAspectRatio
			
			# Set the viewer's scroll bar behaviour.
			#   Qt.ScrollBarAlwaysOff: Never show scroll bar.
			#   Qt.ScrollBarAlwaysOn: Always show scroll bar.
			#   Qt.ScrollBarAsNeeded: Show scroll bar only when zoomed.
			self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
			self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
			
			# Allow zooming with right mouse button.
			# Drag for zoom box, doubleclick to view full image.
			self.view.canZoom = True
			
			# Allow panning with left mouse button.
			self.view.canPan = True
				
			# Load an image to be displayed.
			self.view.loadImageFromFile()  # Pops up file dialog.

			# Handle left mouse clicks with your own custom slot _handleLeftClick(x, y). (x, y) are image coordinates.
			# For (row, col) matrix indexing, row=y and col=x.
			# ImageViewerQt also provides similar signals for left/right mouse button press, release and doubleclick.
			self.view.leftMouseButtonPressed.connect(self._handleLeftClick)

			layout = QVBoxLayout()
			layout.addWidget(self.view)
			self.setLayout(layout)

		# TODO [modify] >> Not correctly working.
		def wheelEvent(self, event):
			self.view.zoom += event.angleDelta().y() / 2880
			self.view.scale(self.view.zoom, self.view.zoom)

		def keyPressEvent(self, event):
			if event.key() == Qt.Key_F11 or event.key() == Qt.Key_F:
				self.toggleFullScreen()

		def toggleFullScreen(self):
			if self.isFullScreen():
				self.showNormal()
			else:
				self.showFullScreen()

		# Custom slot for handling mouse clicks in our viewer.
		# Just prints the (row, column) matrix index of the image pixel that was clicked on.
		def _handleLeftClick(self, x, y):
			row = int(y)
			column = int(x)
			print('Pixel (row = {}, column = {}).'.format(str(column), str(row)))

	#--------------------
	import sys

	print('Using Qt {}.'.format(QT_VERSION_STR))

	app = QApplication(sys.argv)

	widget = MyWidget(None)
	widget.setGeometry(100, 100, 1000, 800)
	widget.show()

	sys.exit(app.exec_())

def main():
	#graphics_view_example()

	#simple_photo_viewer_example()
	simple_image_viewer_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
