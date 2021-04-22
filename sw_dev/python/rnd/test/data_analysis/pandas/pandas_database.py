#!/usr/bin/env python

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.types import Integer

#---------------------------------------------------------------------

# REF [site] >> https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html
def sqlite_example():
	# Create an engine that stores data in the local directory's file.
	#engine = create_engine('sqlite:///sqlite.db')
	# Create an in-memory SQLite database.
	engine = create_engine('sqlite://', encoding='latin1', echo=False)

	df = pd.DataFrame({'name' : ['User 1', 'User 2', 'User 3']})
	print(df)

	df.to_sql('users', con=engine)
	print(engine.execute('SELECT * FROM users').fetchall())

	#--------------------
	df1 = pd.DataFrame({'name' : ['User 4', 'User 5']})

	df1.to_sql('users', con=engine, if_exists='append')
	print(engine.execute('SELECT * FROM users').fetchall())

	df1.to_sql('users', con=engine, if_exists='replace', index_label='id')
	print(engine.execute('SELECT * FROM users').fetchall())

	#--------------------
	# Specify the dtype (especially useful for integers with missing values).

	df = pd.DataFrame({'A': [1, None, 2]})
	print(df)

	df.to_sql('integers', con=engine, index=False, dtype={'A': Integer()})
	print(engine.execute('SELECT * FROM integers').fetchall())

def csv_to_sqlite():
	csv_filepath = './iris.csv'
	db_url = 'sqlite:///iris.db'
	db_table_name = 'iris'

	df = pd.read_csv(csv_filepath, sep=',', header='infer')

	# Create an engine that stores data in the local directory's file.
	engine = create_engine(db_url)

	df.to_sql(db_table_name, con=engine, if_exists='replace', index_label='index')

def main():
	sqlite_example()

	csv_to_sqlite()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
