#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, io, glob, zipfile
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# REF [site] >> https://towardsdatascience.com/a-complete-guide-to-time-series-data-visualization-in-python-da0ddd2cfb01
def example_1():
	df = pd.read_csv("./stock_data.csv", parse_dates=True, index_col="Date")
	if False:
		print(df.head())

		df['Volume'].plot()
		df.plot(subplots=True, figsize=(10, 12))

	df_month = df.resample("M").mean()
	if False:
		fig, ax = plt.subplots(figsize=(10, 6))
		ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m'))
		ax.bar(df_month['2016':].index, df_month.loc['2016':, "Volume"], width=25, align='center')

	df2 = df.reset_index()
	df2['Month'] = df2['Date'].dt.strftime('%b')

	#--------------------
	# Seasonality.
	if False:
		# Start, end = '2016-01', '2016-12'.
		fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
		for name, ax in zip(['Open', 'Close', 'High', 'Low'], axes):
			sns.boxplot(data=df2, x='Month', y=name, ax=ax)
			ax.set_ylabel("")
			ax.set_title(name)
			if ax != axes[-1]:
				ax.set_xlabel('')

	#--------------------
	# Resampling and rolling.
	if False:
		df_month['Volume'].plot(figsize=(8, 6))

		df_week = df.resample("W").mean()

		start, end = '2015-01', '2015-08'
		fig, ax = plt.subplots()
		ax.plot(df.loc[start:end, 'Volume'], marker='.', linestyle='-', linewidth = 0.5, label='Daily', color='black')
		ax.plot(df_week.loc[start:end, 'Volume'], marker='o', markersize=8, linestyle='-', label='Weekly', color='coral')
		ax.set_ylabel("Open")
		ax.legend()

		df_7d_rolling = df.rolling(7, center=True).mean()

		start, end = '2016-06', '2017-05'
		fig, ax = plt.subplots()
		ax.plot(df.loc[start:end, 'Volume'], marker='.', linestyle='-', linewidth=0.5, label='Daily')
		ax.plot(df_week.loc[start:end, 'Volume'], marker='o', markersize=5, linestyle='-', label = 'Weekly mean volume')
		ax.plot(df_7d_rolling.loc[start:end, 'Volume'], marker='.', linestyle='-', label='7d Rolling Average')
		ax.set_ylabel('Stock Volume')
		ax.legend()

	#--------------------
	# Plotting the change.
	if False:
		df['Change'] = df.Close.div(df.Close.shift())
		df['Change'].plot(figsize=(20, 8), fontsize = 16)
		plt.figure()
		df['Change']['2017'].plot(figsize=(10, 6))

	#--------------------
	# Percent change.
	if False:
		df_month.loc[:, 'Percent_Change'] = df.Close.pct_change() * 100
		
		fig, ax = plt.subplots()
		df_month['Percent_Change' ].plot(kind='bar', color='coral', ax=ax)
		ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator())
		ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
		plt.xticks(rotation=45)
		ax.legend()

	#--------------------
	# Differencing.
	if False:
		df.High.diff().plot(figsize=(10, 6))

	#--------------------
	# Expanding window.
	if False:
		fig, ax = plt.subplots()
		ax = df.High.plot(label='High')
		ax = df.High.expanding().mean().plot(label='High expanding mean')
		ax = df.High.expanding().std().plot(label='High expanding std')
		ax.legend()

	#--------------------
	# Heat map.
	if False:
		import calendar

		# Not working.
		all_month_year_df = pd.pivot_table(df2, values="Open", index=["Month"], columns=["Year"], fill_value=0, margins=True)
		named_index = [[calendar.month_abbr[i] if isinstance(i, int) else i for i in list(all_month_year_df.index)]] # name months
		all_month_year_df = all_month_year_df.set_index(named_index)
		print(all_month_year_df)

		ax = sns.heatmap(
			all_month_year_df, cmap='RdYlGn_r', robust=True, fmt='.2f', 
			annot=True, linewidths=.5, annot_kws={'size': 11}, 
			cbar_kws={'shrink': .8, 'label': 'Open'}
		)                       
		ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
		ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
		plt.title('Average Opening', fontdict={'fontsize':18},    pad=14);

	#--------------------
	# Decomposition.
	if True:
		from pylab import rcParams
		import statsmodels.api as sm

		rcParams['figure.figsize'] = 11, 9
		decomposition = sm.tsa.seasonal_decompose(df_month['Volume'], model='Additive')
		fig = decomposition.plot()
		plt.show()

	plt.show()

# REF [site] >> https://towardsdatascience.com/an-ultimate-guide-to-time-series-analysis-in-pandas-76a0433621f3
def example_2():
	if False:
		dates = ['2020-11-25 2:30:00 PM', 'Jan 5, 2020 18:45:00', '01/11/2020', '2020.01.11', '2020/01/11', '20201105']
		print(pd.to_datetime(dates))
		print(pd.to_datetime(dates).strftime('%d-%m-%y'))

	#--------------------
	#df = pd.read_csv('./FB_data.csv')
	df = pd.read_csv('./FB_data.csv', parse_dates=['Date'], index_col="Date")
	if False:
		print(df.head())

	#--------------------
	# Resampling.
	if False:
		print(df.loc["2019-06"])
		print(df.loc["2019-06"].Open.mean())
		print(df.loc["2019-06-21"])
		print(df.loc["2019-06-27":"2019-07-10"])

		print(df.Close.resample('M').mean())
		df.Close.resample('M').mean().plot()
		plt.figure()
		df.Close.resample('W').mean().plot()
		plt.figure()
		df.Close.resample('Q').mean().plot(kind='bar')

	#--------------------
	# Shift.
	if False:
		df1 = pd.DataFrame(df['Open'])
		print(df1.head())
		print(df1.shift(1))
		print(df1.shift(-1))

		df1['Prev Day Opening'] = df1['Open'].shift(1)
		print(df1)

		df1['1 day change'] = df1['Open'] - df1['Prev Day Opening']
		print(df1)

		df1['One week total return'] = (df1['Open'] - df1['Open'].shift(5)) * 100 / df1['Open'].shift(5)
		print(df1.tail())

	#--------------------
	# Timezone.
	if False:
		from pytz import all_timezones
		print(all_timezones)

		df.index = df.index.tz_localize(tz='US/Eastern')
		print(df.index)

		df = df.tz_convert('Europe/Berlin')
		print(df.index)

	#--------------------
	# How to generate missing dates.
	if False:
		rng = pd.date_range(start='11/1/2020', periods=10)
		print(rng)

		# What if I need only business days?
		rng = pd.date_range(start='11/1/2020', periods=10, freq='B')
		print(rng)

	#--------------------
	# Rolling.
	if False:
		print(df[["High"]].rolling(3).mean()[:10])

		data_rol = df[['High', 'Low']].rolling(window=7, center=True).mean()
		print(data_rol)

		import matplotlib.ticker as ticker 
		fig, ax = plt.subplots(figsize= (11, 4))
		ax.plot(df['High'], marker='.', markersize=4, color='0.4', linestyle='None', label='Daily')
		ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
		ax.plot(data_rol['High'], linewidth=2, label='7-d rolling mean')
		ax.set_xlabel('Month')

	#--------------------
	# Differencing for removing the trend.
	if False:
		df_first_order_diff = df[['High', 'Low']].diff()
		#df_first_order_diff = df[['High', 'Low']].diff(3)
		print(df_first_order_diff)

		import matplotlib.ticker as ticker 
		start = '20-06-19'
		fig, ax = plt.subplots(figsize = (11, 4))
		ax.plot(df_first_order_diff.loc[start:, "High"], marker = 'o',  markersize = 4, linestyle = '-', label = 'First Order Differencing')
		ax.xaxis.set_major_locator(ticker.MultipleLocator(30))

	#--------------------
	# Important time feature 3xtraction.
	if False:
		#pd.DatetimeIndex(df.index).year
		#pd.DatetimeIndex(df.index).month
		#pd.DatetimeIndex(df.index).weekday

		df3 = df[['High','Low', 'Volume']]
		df3['Year'] = pd.DatetimeIndex(df3.index).year
		df3['Weekday'] = pd.DatetimeIndex(df3.index).to_series().dt.day_name()
		print(df3)

		fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
		for name, ax in zip(['High', 'Low', 'Volume'], axes):
			sns.boxplot(data=df3, x = 'Weekday', y = name, ax=ax)
			ax.set_title(name)

	#--------------------
	# Period and period index.
	if True:
		y = pd.Period('2020')
		#print(dir(y))

		print(y.start_time)
		print(y.end_time)

		month = pd.Period('2020-2', freq="M") 

		print(month.start_time)
		print(month.end_time)

		print(month + 2)

		d = pd.Period('2020-02-28')
		print(d + 2)

		q = pd.Period('2020Q1')
		print(q.asfreq('M', how='start'))
		print(q.asfreq('M', how='end'))
		print(q + 2)

		q1 = pd.Period('2020Q2', freq='Q-Jan')
		print(q1)

		idx = pd.period_range('2017', '2020', freq='Q')
		print(idx)

		idx = pd.period_range('2017', '2020', freq='Q-Jan')
		print(idx)

		idx = idx.to_timestamp()
		print(idx)

		print(idx.to_period())

	plt.show()

# REF [site] >>
#	https://tomaugspurger.github.io/modern-7-timeseries.html
#	https://github.com/TomAugspurger/effective-pandas/blob/master/modern_7_timeseries.ipynb
def example_3():
	import pandas_datareader as pdr

	gs = pdr.data.DataReader("GS", data_source='yahoo', start='2006-01-01', end='2010-01-01')
	print(gs.head().round(2))
	print(gs.loc[pd.Timestamp('2006-01-01'):pd.Timestamp('2006-12-31')].head())
	print(gs.loc['2006'].head())

	#--------------------
	# Resampling.
	if True:
		print(gs.resample("5d").mean().head())
		print(gs.resample("W").agg(['mean', 'sum']).head())

		# You can up-sample to convert to a higher frequency. The new points are filled with NaNs.
		print(gs.resample("6h").mean().head())

	#--------------------
	# Rolling, expanding, exponential weighted (EW).
	if False:
		gs.Close.plot(label='Raw')
		gs.Close.rolling(28).mean().plot(label='28D MA')
		gs.Close.expanding().mean().plot(label='Expanding Average')
		gs.Close.ewm(alpha=0.03).mean().plot(label='EWMA($\\alpha=.03$)')

		plt.legend(bbox_to_anchor=(1.25, .5))
		plt.tight_layout()
		plt.ylabel("Close ($)")
		sns.despine()

		# Each of .rolling, .expanding, and .ewm return a deferred object, similar to a GroupBy.
		roll = gs.Close.rolling(30, center=True)

		m = roll.agg(['mean', 'std'])
		plt.figure()
		ax = m['mean'].plot()
		ax.fill_between(m.index, m['mean'] - m['std'], m['mean'] + m['std'], alpha=.25)
		plt.tight_layout()
		plt.ylabel("Close ($)")
		sns.despine()

	#--------------------
	# Grab bag.
	if False:
		# Offsets.
		#	These are similar to dateutil.relativedelta, but works with arrays.
		print(gs.index + pd.DateOffset(months=3, days=-2))

		# Holiday calendars.
		from pandas.tseries.holiday import USColumbusDay
		print(USColumbusDay.dates('2015-01-01', '2020-01-01'))

		# Timezones.
		# tz naiive -> tz aware..... to desired UTC
		print(gs.tz_localize('US/Eastern').tz_convert('UTC').head())

	#--------------------
	# Modeling time series.
	if True:
		from collections import namedtuple
		import statsmodels.formula.api as smf
		import statsmodels.tsa.api as smt
		import statsmodels.api as sm
		from modern_pandas_utils import download_timeseries

		def download_many(start, end):
			months = pd.period_range(start, end=end, freq='M')
			# We could easily parallelize this loop.
			for i, month in enumerate(months):
				download_timeseries(month)

		def time_to_datetime(df, columns):
			'''
			Combine all time items into datetimes.
			2014-01-01,1149.0 -> 2014-01-01T11:49:00
			'''
			def converter(col):
				timepart = (col.astype(str)
					.str.replace('\.0$', '')  # NaNs force float dtype
					.str.pad(4, fillchar='0'))
				return  pd.to_datetime(df['fl_date'] + ' ' + timepart.str.slice(0, 2) + ':' + timepart.str.slice(2, 4), errors='coerce')
				return datetime_part
			df[columns] = df[columns].apply(converter)
			return df

		def unzip_one(fp):
			try:
				zf = zipfile.ZipFile(fp)
				csv = zf.extract(zf.filelist[0])
				return csv
			except zipfile.BadZipFile as ex:
				print('zipfile.BadZipFile raised in {}: {}.'.format(fp, ex))
				raise

		def read_one(fp):
			df = (pd.read_csv(fp, encoding='latin1')
				.rename(columns=str.lower)
				.drop('unnamed: 6', axis=1)
				.pipe(time_to_datetime, ['dep_time', 'arr_time', 'crs_arr_time', 'crs_dep_time'])
				.assign(fl_date=lambda x: pd.to_datetime(x['fl_date'])))
			return df

		store = './modern_pandas_data/ts.hdf5'

		if not os.path.exists(store):
			download_many('2000-01-01', '2016-01-01')

			zips = glob.glob(os.path.join('modern_pandas_data', 'timeseries', '*.zip'))
			csvs = [unzip_one(fp) for fp in zips]
			dfs = [read_one(fp) for fp in csvs]
			df = pd.concat(dfs, ignore_index=True)

			df['origin'] = df['origin'].astype('category')
			df.to_hdf(store, 'ts', format='table')
		else:
			df = pd.read_hdf(store, 'ts')

		with pd.option_context('display.max_rows', 100):
			print(df.dtypes)

		daily = df.fl_date.value_counts().sort_index()
		y = daily.resample('MS').mean()
		print(y.head())

		ax = y.plot()
		ax.set(ylabel='Average Monthly Flights')
		sns.despine()

		X = (pd.concat([y.shift(i) for i in range(6)], axis=1, keys=['y'] + ['L%s' % i for i in range(1, 6)]).dropna())
		print(X.head())

		mod_lagged = smf.ols('y ~ trend + L1 + L2 + L3 + L4 + L5', data=X.assign(trend=np.arange(len(X))))
		res_lagged = mod_lagged.fit()
		res_lagged.summary()

		sns.heatmap(X.corr())

		ax = res_lagged.params.drop(['Intercept', 'trend']).plot.bar(rot=0)
		plt.ylabel('Coefficeint')
		sns.despine()

		# Autocorrelation.
		# 'Results.resid' is a series of residuals: y - ŷ.
		mod_trend = sm.OLS.from_formula('y ~ trend', data=y.to_frame(name='y').assign(trend=np.arange(len(y))))
		res_trend = mod_trend.fit()

		def tsplot(y, lags=None, figsize=(10, 8)):
			fig = plt.figure(figsize=figsize)
			layout = (2, 2)
			ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
			acf_ax = plt.subplot2grid(layout, (1, 0))
			pacf_ax = plt.subplot2grid(layout, (1, 1))
			
			y.plot(ax=ts_ax)
			smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
			smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
			[ax.set_xlim(1.5) for ax in [acf_ax, pacf_ax]]
			sns.despine()
			plt.tight_layout()
			return ts_ax, acf_ax, pacf_ax

		tsplot(res_trend.resid, lags=36)

		y.to_frame(name='y').assign(Δy=lambda x: x.y.diff()).plot(subplots=True)
		sns.despine()

		ADF = namedtuple("ADF", "adf pvalue usedlag nobs critical icbest")

		#ADF(*smt.adfuller(y))._asdict()
		ADF(*smt.adfuller(y.dropna()))._asdict()
		ADF(*smt.adfuller(y.diff().dropna()))._asdict()

		data = (y.to_frame(name='y').assign(Δy=lambda df: df.y.diff()).assign(LΔy=lambda df: df.Δy.shift()))
		mod_stationary = smf.ols('Δy ~ LΔy', data=data.dropna())
		res_stationary = mod_stationary.fit()

		tsplot(res_stationary.resid, lags=24)

		# Seasonality.
		#smt.seasonal_decompose(y).plot()
		smt.seasonal_decompose(y.fillna(method='ffill')).plot()

		# ARIMA.
		mod = smt.SARIMAX(y, trend='c', order=(1, 1, 1))
		res = mod.fit()
		tsplot(res.resid[2:], lags=24)

		res.summary()

		mod_seasonal = smt.SARIMAX(y, trend='c', order=(1, 1, 2), seasonal_order=(0, 1, 2, 12), simple_differencing=False)
		res_seasonal = mod_seasonal.fit()

		res_seasonal.summary()

		tsplot(res_seasonal.resid[12:], lags=24)

		# Forecasting.
		pred = res_seasonal.get_prediction(start='2001-03-01')
		pred_ci = pred.conf_int()

		plt.figure()
		ax = y.plot(label='observed')
		pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7)
		ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
		ax.set_ylabel("Monthly Flights")
		plt.legend()
		sns.despine()

		pred_dy = res_seasonal.get_prediction(start='2002-03-01', dynamic='2013-01-01')
		pred_dy_ci = pred_dy.conf_int()

		plt.figure()
		ax = y.plot(label='observed')
		pred_dy.predicted_mean.plot(ax=ax, label='Forecast')
		ax.fill_between(pred_dy_ci.index, pred_dy_ci.iloc[:, 0], pred_dy_ci.iloc[:, 1], color='k', alpha=.25)
		ax.set_ylabel("Monthly Flights")

		# Highlight the forecast area.
		ax.fill_betweenx(ax.get_ylim(), pd.Timestamp('2013-01-01'), y.index[-1], alpha=.1, zorder=-1)
		ax.annotate('Dynamic $\\longrightarrow$', (pd.Timestamp('2013-02-01'), 550))

		plt.legend()
		sns.despine()

	plt.show()

def main():
	#example_1()
	#example_2()
	example_3()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
