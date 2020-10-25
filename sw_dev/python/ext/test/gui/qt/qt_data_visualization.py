#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys
import pandas as pd
from PySide2.QtCore import QDateTime, QTimeZone
from PySide2.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide2.QtCore import Slot
from PySide2.QtGui import QColor, QPainter, QKeySequence
from PySide2.QtWidgets import QApplication, QMainWindow, QAction
from PySide2.QtWidgets import QWidget, QHBoxLayout, QHeaderView, QTableView, QSizePolicy
from PySide2.QtCharts import QtCharts

class CustomTableModel(QAbstractTableModel):
	def __init__(self, data=None):
		super().__init__()
		self.load_data(data)

	def load_data(self, data):
		self.input_dates = data[0].values
		self.input_magnitudes = data[1].values

		self.column_count = 2
		self.row_count = len(self.input_magnitudes)

	def rowCount(self, parent=QModelIndex()):
		return self.row_count

	def columnCount(self, parent=QModelIndex()):
		return self.column_count

	def headerData(self, section, orientation, role):
		if role != Qt.DisplayRole:
			return None
		if orientation == Qt.Horizontal:
			return ('Date', 'Magnitude')[section]
		else:
			return '{}'.format(section)

	def data(self, index, role=Qt.DisplayRole):
		column = index.column()
		row = index.row()

		if role == Qt.DisplayRole:
			if column == 0:
				raw_date = self.input_dates[row]
				date = '{}'.format(raw_date.toPython())
				return date[:-3]
			elif column == 1:
				return '{:.2f}'.format(self.input_magnitudes[row])
		elif role == Qt.BackgroundRole:
			return QColor(Qt.white)
		elif role == Qt.TextAlignmentRole:
			return Qt.AlignRight

		return None

class MyWidget(QWidget):
	def __init__(self, data):
		super().__init__()

		# Get the model.
		self.model = CustomTableModel(data)

		# Create a QTableView.
		self.table_view = QTableView()
		self.table_view.setModel(self.model)

		# QTableView headers.
		self.horizontal_header = self.table_view.horizontalHeader()
		self.vertical_header = self.table_view.verticalHeader()
		self.horizontal_header.setSectionResizeMode(QHeaderView.ResizeToContents)
		self.vertical_header.setSectionResizeMode(QHeaderView.ResizeToContents)
		self.horizontal_header.setStretchLastSection(True)

		# Create QChart.
		self.chart = QtCharts.QChart()
		self.chart.setAnimationOptions(QtCharts.QChart.AllAnimations)
		self.add_series('Magnitude (Column 1)', [0, 1])

		# Create QChartView.
		self.chart_view = QtCharts.QChartView(self.chart)
		self.chart_view.setRenderHint(QPainter.Antialiasing)

		# QWidget layout.
		self.main_layout = QHBoxLayout()
		size = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

		# Left layout.
		size.setHorizontalStretch(1)
		self.table_view.setSizePolicy(size)
		self.main_layout.addWidget(self.table_view)

		# Right layout.
		size.setHorizontalStretch(4)
		self.chart_view.setSizePolicy(size)
		self.main_layout.addWidget(self.chart_view)

		# Set the layout to the QWidget.
		self.setLayout(self.main_layout)

	def add_series(self, name, columns):
		# Create QLineSeries.
		self.series = QtCharts.QLineSeries()
		self.series.setName(name)

		# Fill QLineSeries.
		for i in range(self.model.rowCount()):
			# Get the data.
			t = self.model.index(i, 0).data()
			date_fmt = 'yyyy-MM-dd HH:mm:ss.zzz'

			x = QDateTime().fromString(t, date_fmt).toSecsSinceEpoch()
			y = float(self.model.index(i, 1).data())

			if x > 0 and y > 0:
				self.series.append(x, y)

		self.chart.addSeries(self.series)

		# Set X-axis.
		self.axis_x = QtCharts.QDateTimeAxis()
		self.axis_x.setTickCount(10)
		self.axis_x.setFormat('dd.MM (h:mm)')
		self.axis_x.setTitleText('Date')
		self.chart.addAxis(self.axis_x, Qt.AlignBottom)
		self.series.attachAxis(self.axis_x)
		# Set Y-axis.
		self.axis_y = QtCharts.QValueAxis()
		self.axis_y.setTickCount(10)
		self.axis_y.setLabelFormat('%.2f')
		self.axis_y.setTitleText('Magnitude')
		self.chart.addAxis(self.axis_y, Qt.AlignLeft)
		self.series.attachAxis(self.axis_y)

		# Get the color from the QChart to use it on the QTableView.
		self.model.color = '{}'.format(self.series.pen().color().name())

class MyMainWindow(QMainWindow):
	def __init__(self, widget):
		super().__init__()
		self.setWindowTitle('Eartquakes information')
		self.setCentralWidget(widget)

		# Menu.
		self.menu = self.menuBar()
		self.file_menu = self.menu.addMenu('File')

		# Exit QAction.
		exit_action = QAction('Exit', self)
		exit_action.setShortcut(QKeySequence.Quit)
		exit_action.triggered.connect(self.close)

		self.file_menu.addAction(exit_action)

		# Status bar.
		self.status = self.statusBar()
		self.status.showMessage('Data loaded and plotted')

		# Window dimensions.
		geometry = QApplication.instance().desktop().availableGeometry(self)
		self.setFixedSize(geometry.width() * 0.8, geometry.height() * 0.7)

def transform_date(utc, timezone=None):
	utc_fmt = 'yyyy-MM-ddTHH:mm:ss.zzzZ'
	new_date = QDateTime().fromString(utc, utc_fmt)
	if timezone:
		new_date.setTimeZone(timezone)
	return new_date

def read_data(csv_filepath):
	# Read the CSV content.
	df = pd.read_csv(csv_filepath)

	# Remove wrong magnitudes.
	df = df.drop(df[df.mag < 0].index)
	magnitudes = df['mag']

	# My local timezone.
	timezone = QTimeZone(b'Europe/Berlin')

	# Get timestamp transformed to our timezone.
	times = df['time'].apply(lambda x: transform_date(x, timezone))

	return times, magnitudes

def main():
	# REF [site] >> https://doc.qt.io/qtforpython/tutorials/datavisualize/index.html

	csv_filepath = './all_hour.csv'
	data = read_data(csv_filepath)
	#print(data)

	app = QApplication(sys.argv)

	widget = MyWidget(data)
	window = MyMainWindow(widget)
	window.show()

	sys.exit(app.exec_())

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
