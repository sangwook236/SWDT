#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from datetime import datetime, date, time

# REF [site] >> https://docs.python.org/3/library/datetime.html

def simple_example():
	#--------------------
	# Datetime.

	datetime.fromisoformat('2016-04-15')
	#datetime.fromisoformat('16-4-15')  # ValueError: Invalid isoformat string.
	#datetime.fromisoformat('16-04-15')  # ValueError: Invalid isoformat string.
	#datetime.fromisoformat('2016-4-15')  # ValueError: Invalid isoformat string.
	#datetime.fromisoformat('2016.04.15')  # ValueError: Invalid isoformat string.

	datetime.fromisoformat('2021-07-22T10:46:00.725476')
	datetime.fromisoformat('2021-07-22T10:46:00')
	datetime.fromisoformat('2021-07-22 10:46:00.725476')
	datetime.fromisoformat('2021-07-22 10:46:00')

	datetime.strptime('16/4/15', '%y/%m/%d')
	datetime.strptime('2016/4/15', '%Y/%m/%d')
	datetime.strptime('16-4-15', '%y-%m-%d')
	datetime.strptime('2016-4-15', '%Y-%m-%d')
	datetime.strptime('16.4.15', '%y.%m.%d')
	datetime.strptime('2016.4.15', '%Y.%m.%d')
	datetime.strptime('15/4/2016', '%d/%m/%Y')
	datetime.strptime('15/4/16', '%d/%m/%y')
	datetime.strptime('15/4/2016', '%d/%m/%Y')
	datetime.strptime('15-4-16', '%d-%m-%y')
	datetime.strptime('15-4-2016', '%d-%m-%Y')
	datetime.strptime('4/15/16', '%m/%d/%y')
	datetime.strptime('4/15/2016', '%m/%d/%Y')
	datetime.strptime('4-15-16', '%m-%d-%y')
	datetime.strptime('4-15-2016', '%m-%d-%Y')

	datetime.strptime('4 15, 2016', '%m %d, %Y')
	datetime.strptime('April 15, 16', '%B %d, %y')
	datetime.strptime('April 15, 2016', '%B %d, %Y')
	datetime.strptime('Apr 15, 2016', '%b %d, %Y')

	#datetime.fromtimestamp(timestamp, tz=None)
	#datetime.utcfromtimestamp(timestamp)
	#datetime.fromordinal(ordinal)
	#datetime.fromisocalendar(year, week, day)

	print('Datetime: min = {}, max = {}, resolution = {}.'.format(datetime.min, datetime.max, datetime.resolution))

	now = datetime.now(tz=None)
	utcnow = datetime.utcnow()

	print('Now: {}.'.format(now.isoformat(sep='T', timespec='auto')))
	print('UTC Now: {}.'.format(utcnow.isoformat(sep='T', timespec='auto')))

	print('Now (strftime): {}.'.format(now.strftime('%Y-%m-%dT%H:%M:%S.%f')))
	print('Now (strftime): {}.'.format(now.strftime('%B %d, %y')))
	print('Now (strftime): {}.'.format(now.strftime('%B %d, %Y')))

	print('Now: year = {}, month = {}, day = {}, hour = {}, minute = {}, second = {}, microsecond = {}.'.format(now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond))
	print('Now: tzinfo = {}.'.format(now.tzinfo))
	print('Now: fold = {}.'.format(now.fold))

	print('Now: weekday = {}, isoweekday = {}, isocalendar = {}.'.format(now.weekday(), now.isoweekday(), now.isocalendar()))
	print('Now: timetuple = {}, utctimetuple = {}.'.format(now.timetuple(), now.utctimetuple()))
	print('Now: timestamp = {}.'.format(now.timestamp()))
	print('Now: tzname = {}, dst = {}, utcoffset = {}.'.format(now.tzname(), now.dst(), now.utcoffset()))
	print('Now: astimezone = {}, ctime = {}.'.format(now.astimezone(tz=None), now.ctime()))
	print('Now: toordinal = {}.'.format(now.toordinal()))
	#now.replace(year=self.year, month=self.month, day=self.day, hour=self.hour, minute=self.minute, second=self.second, microsecond=self.microsecond, tzinfo=self.tzinfo, *, fold=0)

	#--------------------
	# Time.

	time.fromisoformat('01:23:45.098765')
	time.fromisoformat('01:23:45')
	#time.fromisoformat('1:23:45')  # ValueError: Invalid isoformat string.

	print('Time: min = {}, max = {}, resolution = {}.'.format(time.min, time.max, time.resolution))

	now_time = datetime.now(tz=None).time()
	#now_time = datetime.now(tz=None).timetz()
	print('Now (time): {}.'.format(now_time.isoformat(timespec='auto')))
	print('Now (time): {}.'.format(now_time.strftime('%H=%M=%S.%f')))

	print('Now (time): hour = {}, minute = {}, second = {}, microsecond = {}.'.format(now_time.hour, now_time.minute, now_time.second, now_time.microsecond))
	print('Now (time): tzinfo = {}.'.format(now_time.tzinfo))
	print('Now (time): fold = {}.'.format(now_time.fold))
	print('Now: tzname = {}, dst = {}, utcoffset = {}.'.format(now.tzname(), now.dst(), now.utcoffset()))
	#now_time.replace(hour=self.hour, minute=self.minute, second=self.second, microsecond=self.microsecond, tzinfo=self.tzinfo, *, fold=0)

	#--------------------
	# Date.

	date.fromisoformat('2021-07-13')
	#date.fromisoformat('2021-7-13')  # ValueError: Invalid isoformat string.
	#date.fromisoformat('21-7-13')  # ValueError: Invalid isoformat string.

	#date.fromtimestamp(timestamp)
	#date.fromordinal(ordinal)
	#date.fromisocalendar(year, week, day)

	print('Date: min = {}, max = {}, resolution = {}.'.format(date.min, date.max, date.resolution))

	today = date.today()
	print('Today: {}.'.format(today.isoformat()))

	now_date = datetime.now(tz=None).date()
	print('Now (date): {}.'.format(now_date.isoformat()))
	print('Now (date): {}.'.format(now_date.strftime('%B %d, %y')))

	print('Now (date): year = {}, month = {}, day = {}.'.format(now_date.year, now_date.month, now_date.day))
	#print('Now (date): tzinfo = {}.'.format(now_date.tzinfo))
	#print('Now (date): fold = {}.'.format(now_date.fold))

	print('Now (date): weekday = {}, isoweekday = {}, isocalendar = {}.'.format(now_date.weekday(), now_date.isoweekday(), now_date.isocalendar()))
	print('Now (date): timetuple = {}.'.format(now_date.timetuple()))
	print('Now (date): ctime = {}.'.format(now_date.ctime()))
	print('Now (date): toordinal = {}.'.format(now_date.toordinal()))
	#now_date.replace(year=self.year, month=self.month, day=self.day)

def main():
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
