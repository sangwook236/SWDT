#!/usr/bin/env python

import sqlite3 as lite
import sys

def check_version():
	connection = None

	try:
		connection = lite.connect('test.db')
		cur = connection.cursor()    
		cur.execute('SELECT SQLITE_VERSION()')
		data = cur.fetchone()
		print('SQLite version: {}.'.format(data))
	except lite.Error as ex:   
		print('Error: {}.'.format(ex.args[0]))
		sys.exit(1)
	finally:
		if connection:
			connection.close()

def create_and_insert():
	with lite.connect('user.db') as connection:
		try:
			cur = connection.cursor()
			cur.execute('CREATE TABLE Users(Id INT, Name TEXT)')

			cur.execute('INSERT INTO Users VALUES(1, "Michelle")')
			cur.execute('INSERT INTO Users VALUES(2, "Sonya")')
			cur.execute('INSERT INTO Users VALUES(3, "Greg")')
		except lite.Error as ex:   
			print('Error: {}.'.format(ex.args[0]))
			sys.exit(1)

def query():
	with lite.connect('user.db') as connection:
		try:
			cur = connection.cursor()
			cur.execute('SELECT * FROM Users')

			rows = cur.fetchall()

			for row in rows:
				print(row)
		except lite.Error as ex:
			print('Error: {}.'.format(ex.args[0]))
			sys.exit(1)

def main():
	check_version()
	#create_and_insert()
	query()

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
