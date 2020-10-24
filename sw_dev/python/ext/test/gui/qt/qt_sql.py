#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import datetime, logging
from PySide2.QtCore import Qt, Slot, QDir, QFile, QUrl
from PySide2.QtSql import QSqlDatabase, QSqlQuery, QSqlRecord, QSqlTableModel
from PySide2.QtGui import QGuiApplication
from PySide2.QtQml import QQmlApplicationEngine

table_name = 'Conversations'

def createTable():
	if table_name in QSqlDatabase.database().tables():
		return

	query = QSqlQuery()
	if not query.exec_(
		"""
		CREATE TABLE IF NOT EXISTS 'Conversations' (
			'author' TEXT NOT NULL,
			'recipient' TEXT NOT NULL,
			'timestamp' TEXT NOT NULL,
			'message' TEXT NOT NULL,
		FOREIGN KEY('author') REFERENCES Contacts ( name ),
		FOREIGN KEY('recipient') REFERENCES Contacts ( name )
		)
		"""
	):
		logging.error("Failed to query database")

	# This adds the first message from the Bot
	# and further development is required to make it interactive.
	query.exec_(
		"""
		INSERT INTO Conversations VALUES(
			'machine', 'Me', '2019-01-07T14:36:06', 'Hello!'
		)
		"""
	)
	logging.info(query)

class SqlConversationModel(QSqlTableModel):
	def __init__(self, parent=None):
		super(SqlConversationModel, self).__init__(parent)

		createTable()
		self.setTable(table_name)
		self.setSort(2, Qt.DescendingOrder)
		self.setEditStrategy(QSqlTableModel.OnManualSubmit)
		self.recipient = ''

		self.select()
		logging.debug('Table was loaded successfully.')

	def setRecipient(self, recipient):
		if recipient == self.recipient:
			pass

		self.recipient = recipient

		filter_str = (
			"(recipient = '{}' AND author = 'Me') OR " "(recipient = 'Me' AND author='{}')"
		).format(self.recipient)
		self.setFilter(filter_str)
		self.select()

	def data(self, index, role):
		if role < Qt.UserRole:
			return QSqlTableModel.data(self, index, role)

		sql_record = QSqlRecord()
		sql_record = self.record(index.row())

		return sql_record.value(role - Qt.UserRole)

	def roleNames(self):
		"""Converts dict to hash because that's the result expected
		by QSqlTableModel"""
		names = {}
		author = 'author'.encode()
		recipient = 'recipient'.encode()
		timestamp = 'timestamp'.encode()
		message = 'message'.encode()

		names[hash(Qt.UserRole)] = author
		names[hash(Qt.UserRole + 1)] = recipient
		names[hash(Qt.UserRole + 2)] = timestamp
		names[hash(Qt.UserRole + 3)] = message

		return names

	@Slot(str, str, str)
	def send_message(self, recipient, message, author):
		timestamp = datetime.datetime.now()

		new_record = self.record()
		new_record.setValue('author', author)
		new_record.setValue('recipient', recipient)
		new_record.setValue('timestamp', str(timestamp))
		new_record.setValue('message', message)

		logging.debug('Message: "{}" \n Received by: "{}"'.format(message, recipient))

		if not self.insertRecord(self.rowCount(), new_record):
			logging.error('Failed to send message: {}'.format(self.lastError().text()))
			return

		self.submitAll()
		self.select()

def connectToDatabase(logger):
	database = QSqlDatabase.database()
	if not database.isValid():
		database = QSqlDatabase.addDatabase('QSQLITE')
		if not database.isValid():
			logger.error('Cannot add database')

	write_dir = QDir()
	if not write_dir.mkpath('.'):
		logger.error('Failed to create writable directory')

	# Ensure that we have a writable location on all devices.
	filename = '{}/chat-database.sqlite3'.format(write_dir.absolutePath())

	# When using the SQLite driver, open() will create the SQLite database if it doesn't exist.
	database.setDatabaseName(filename)
	if not database.open():
		logger.error('Cannot open database')
		QFile.remove(filename)

def main():
	# REF [site] >> https://doc.qt.io/qtforpython/tutorials/qmlsqlintegration/qmlsqlintegration.html

	logging.basicConfig(filename='./chat.log', level=logging.DEBUG)
	logger = logging.getLogger('logger')

	app = QGuiApplication()

	connectToDatabase(logger)
	sql_conversation_model = SqlConversationModel()

	engine = QQmlApplicationEngine()
	# Export pertinent objects to QML.
	engine.rootContext().setContextProperty('chat_model', sql_conversation_model)  # chat_model is in ./chat.qml.
	engine.load(QUrl('./chat.qml'))

	app.exec_()

#--------------------------------------------------------------------

# Usage:
#	python qt_sql.py
#	QT_XCB_GL_INTEGRATION=none python qt_sql.py

if '__main__' == __name__:
	main()
