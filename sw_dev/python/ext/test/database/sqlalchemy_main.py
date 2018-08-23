#!/usr/bin/env python

import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class Person(Base):
	__tablename__ = 'person'
	# Here we define columns for the table person.
	# Notice that each column is also a normal Python instance attribute.
	id = Column(Integer, primary_key=True)
	name = Column(String(250), nullable=False)

class Address(Base):
	__tablename__ = 'address'
	# Here we define columns for the table address.
	# Notice that each column is also a normal Python instance attribute.
	id = Column(Integer, primary_key=True)
	street_name = Column(String(250))
	street_number = Column(String(250))
	post_code = Column(String(250), nullable=False)
	person_id = Column(Integer, ForeignKey('person.id'))
	person = relationship(Person)

# REF [site] >> https://www.pythoncentral.io/introductory-tutorial-python-sqlalchemy/
def sqlite_example():
	# Create an engine that stores data in the local directory's file.
	engine = create_engine('sqlite:///sqlalchemy_example.db')
	# Create an in-memory SQLite database.
	#engine = create_engine('sqlite://', echo=False)

	# Create all tables in the engine.
	# This is equivalent to "Create Table" statements in raw SQL.
	Base.metadata.create_all(engine)

	# Bind the engine to the metadata of the Base class so that the declaratives can be accessed through a DBSession instance.
	Base.metadata.bind = engine

	DBSession = sessionmaker(bind=engine)
	# A DBSession() instance establishes all conversations with the database and represents a "staging zone" for all the objects loaded into the database session object.
	# Any change made against the objects in the session won't be persisted into the database until you call session.commit().
	# If you're not happy about the changes, you can revert all of them back to the last commit by calling session.rollback()
	session = DBSession()

	# Insert a Person in the person table.
	new_person = Person(name='new person')
	session.add(new_person)
	session.commit()

	# Insert an Address in the address table.
	new_address = Address(post_code='00000', person=new_person)
	session.add(new_address)
	session.commit()

	people = session.query(Person).all()
	print('#people =', len(people))

	person = session.query(Person).first()
	print('Person name =', person.name)

	addresses = session.query(Address).filter(Address.person == person).all()
	print('#addresses =', len(addresses))

	address = session.query(Address).filter(Address.person == person).one()
	print('Postal code =', address.post_code)

def pandas_example():
	# REF [python] >> ${SWDT_PYTHON_HOME}/rnd/test/data_analysis/pandas/pandas_database.py
	pass

def main():
	sqlite_example()

	pandas_example()  # Not implemented.

#%%------------------------------------------------------------------

if '__main__' == __name__:
	main()
