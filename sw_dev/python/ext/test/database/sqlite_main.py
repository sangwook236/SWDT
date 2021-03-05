#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#import sys
import sqlite3

def check_version():
	try:
		#connection = sqlite3.connect('./sqlite_test.db')  # File DB. Creates a DB file.
		connection = sqlite3.connect(':memory:')  # In-memory DB.
		#connection = sqlite3.connect('file::memory:?cache=shared')  # Shared in-memory DB.
		#connection = sqlite3.connect('file:dbname?mode=memory&cache=shared')  # Named, shared in-memory DB.
		cursor = connection.cursor()    

		cursor.execute('SELECT SQLITE_VERSION()')
		data = cursor.fetchone()
		print('SQLite version: {}.'.format(data))

	except sqlite3.Error as ex:   
		print('sqlite3.Error: {}.'.format(ex))
		#sys.exit(1)
	finally:
		cursor.close()
		if connection:
			connection.close()

def create_and_insert_example():
	def load_image(image_filepath):
		try:
			with open(image_filepath, 'rb') as fd:
				img = fd.read()
				return img
		except IOError as ex:
			print('IOError: {}.'.format(ex))
			return None

	db_filepath = './sqlite_user.db'
	with sqlite3.connect(db_filepath) as connection:  # Creates a DB file.
		try:
			cursor = connection.cursor()

			#cursor.execute('CREATE TABLE Users(Id INTEGER PRIMARY KEY, Name TEXT NOT NULL, Height REAL, MyID INT UNIQUE, Image BLOB)')
			cursor.execute('CREATE TABLE IF NOT EXISTS Users (Id INTEGER PRIMARY KEY, Name TEXT NOT NULL, Height REAL, MyID INT UNIQUE, Image BLOB)')

			cursor.execute('INSERT INTO Users(Name, Height, MyID) VALUES ("Michelle", 175.9, 1)')
			cursor.execute('INSERT INTO Users(Name, Height, MyID) VALUES ("Sonya", 163.7, 2)')
			cursor.execute('INSERT INTO Users(Name, Height, MyID) VALUES (?, ?, ?)', ('Greg', 186.2, 3))

			print('Last row ID = {}.'.format(cursor.lastrowid))
			print('#rows affected = {}.'.format(cursor.rowcount))

			sql = 'INSERT INTO Users(Name, Height, MyID) VALUES (?, ?, ?)'
			cursor.execute(sql, ('Brian', 171.4, 4))

			table_rows = [('Lucy', 169.6, 5), ('Ryan', 176.8, 6)]
			cursor.executemany(sql, table_rows)

			print('Last row ID = {}.'.format(cursor.lastrowid))
			print('#rows affected = {}.'.format(cursor.rowcount))

			# Image.
			imge_filepath = '/path/to/image'
			img = load_image(imge_filepath)  # An object of type 'bytes'.
			if img:
				blob = sqlite3.Binary(img)
				cursor.execute('INSERT INTO Users(Name, Height, MyID, Image) VALUES (?, ?, ?, ?)', ('Michael', 193.5, 7, blob))
			else:
				cursor.execute('INSERT INTO Users(Name, Height, MyID) VALUES (?, ?, ?)', ('Michael', 193.5, 7))

			print('Last row ID = {}.'.format(cursor.lastrowid))
			print('#rows affected = {}.'.format(cursor.rowcount))

			connection.commit()
		except sqlite3.Error as ex:
			connection.rollback()
			print('sqlite3.Error: {}.'.format(ex))
			#sys.exit(1)
		finally:
			cursor.close()

def query_example():
	db_filepath = './sqlite_user.db'
	with sqlite3.connect(db_filepath) as connection:  # Creates a DB file.
		try:
			cursor = connection.cursor()
			#sql = 'SELECT * FROM Users'
			sql = 'SELECT Name, Height, MyID FROM Users'

			print('Rows: {}.'.format([row for row in cursor.execute(sql)]))

			cursor.execute(sql)
			#rows = cursor.fetchall()
			rows = cursor.fetchmany(5)
			print('Rows: {}.'.format(rows))

			cursor.execute(sql)
			while True:
				row = cursor.fetchone()
				if row is None: break
				print(row)

			#--------------------
			cursor.execute('SELECT * FROM Users WHERE Height >= ? LIMIT 1 OFFSET ?', (170, 2))  # Zero-based offset.
			selected_user = cursor.fetchone()  # Returns a single row.
			print('Selected user = {}.'.format(selected_user))

			cursor.execute('SELECT Count(*) FROM Users WHERE Height >= ?', (170,))
			num_users = cursor.fetchone()[0]
			cursor.execute('SELECT AVG(Height) FROM Users WHERE Height >= ?', (180,))
			avg_selected = cursor.fetchone()[0]
			cursor.execute('SELECT SUM(Height) FROM Users WHERE Height >= ?', (180,))
			sum_selected = cursor.fetchone()[0]
			print('#users = {}, average = {}, sum = {}.'.format(num_users, avg_selected, sum_selected))
		except sqlite3.Error as ex:
			print('sqlite3.Error: {}.'.format(ex))
			#sys.exit(1)
		finally:
			cursor.close()

def update_example():
	db_filepath = './sqlite_user.db'
	with sqlite3.connect(db_filepath) as connection:  # Creates a DB file.
		try:
			cursor = connection.cursor()

			#cursor.execute('UPDATE Users SET Height=? WHERE Id=?', (165.8, 4))
			cursor.execute('UPDATE Users SET Height=? WHERE Name=?', (165.8, 'Brian'))

			print('Last row ID = {}.'.format(cursor.lastrowid))
			print('#rows affected = {}.'.format(cursor.rowcount))

			#print('Rows: {}.'.format([row for row in cursor.execute('SELECT * FROM Users')]))
			print('Rows: {}.'.format([row for row in cursor.execute('SELECT Name, Height, MyID FROM Users')]))

			connection.commit()
		except sqlite3.Error as ex:
			connection.rollback()
			print('sqlite3.Error: {}.'.format(ex))
			#sys.exit(1)
		finally:
			cursor.close()

def main():
	check_version()

	create_and_insert_example()
	query_example()
	update_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
