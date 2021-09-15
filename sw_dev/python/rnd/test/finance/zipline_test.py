#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os, datetime
import pandas as pd
import pytz
import zipline
import trading_calendars

# Dual moving average algorithm.
#	REF [site] >> https://github.com/quantopian/zipline
def quickstart_tutorial():
	def initialize(context):
		context.i = 0
		context.asset = zipline.api.symbol("AAPL")

	def handle_data(context, data):
		# Skip first 300 days to get full windows.
		context.i += 1
		if context.i < 300:
			return

		# Compute averages.
		# data.history() has to be called with the same params from above and returns a pandas dataframe.
		short_mavg = data.history(context.asset, "price", bar_count=100, frequency="1d").mean()
		long_mavg = data.history(context.asset, "price", bar_count=300, frequency="1d").mean()

		# Trading logic.
		if short_mavg > long_mavg:
			# order_target orders as many shares as needed to achieve the desired number of shares.
			zipline.api.order_target(context.asset, 100)
		elif short_mavg < long_mavg:
			zipline.api.order_target(context.asset, 0)

		# Save values for later inspection.
		zipline.api.record(
			AAPL=data.current(context.asset, "price"),
			short_mavg=short_mavg,
			long_mavg=long_mavg
		)

	zipline.run_algorithm(
		start=pd.Timestamp("2014-01-01", tz="utc"),
		end=pd.Timestamp("2018-01-01", tz="utc"),
		initialize=initialize,
		capital_base=1e7,
		handle_data=handle_data,
		before_trading_start=None,
		analyze=None,
		data_frequency="daily",
		bundle="quantopian-quandl",  # zipline ingest -b quantopian-quandl
		bundle_timestamp=None,
		trading_calendar=None,
		metrics_set="default",
		benchmark_returns=None,
		default_extension=True,
		extensions=(),
		strict_extensions=True,
		environ=os.environ,
		blotter="default"
	)

# REF [site] >> https://github.com/quantopian/zipline/blob/master/zipline/examples/buyapple.py
def buyapple_example():
	trading_calendars.register_calendar("YAHOO", trading_calendars.get_calendar("NYSE"), force=True)

	def initialize(context):
		context.asset = zipline.api.symbol("AAPL")

		# Explicitly set the commission/slippage to the "old" value until we can rebuild example data.
		# https://github.com/quantopian/zipline/blob/master/tests/resources/rebuild_example_data#L105
		context.set_commission(zipline.finance.commission.PerShare(cost=.0075, min_trade_cost=1.0))
		context.set_slippage(zipline.finance.slippage.VolumeShareSlippage())

	def handle_data(context, data):
		zipline.api.order(context.asset, 10)
		zipline.api.record(AAPL=data.current(context.asset, "price"))

	# Note: this function can be removed if running this algorithm on quantopian.com.
	def analyze(context=None, results=None):
		import matplotlib.pyplot as plt

		# Plot the portfolio and asset data.
		ax1 = plt.subplot(211)
		results.portfolio_value.plot(ax=ax1)
		ax1.set_ylabel("Portfolio value (USD)")
		ax2 = plt.subplot(212, sharex=ax1)
		results.AAPL.plot(ax=ax2)
		ax2.set_ylabel("AAPL price (USD)")

		# Show the plot.
		plt.gcf().set_size_inches(18, 8)
		plt.show()

	start = datetime.datetime(2014, 1, 1, 0, 0, 0, 0, datetime.timezone.utc).date()
	#start = datetime.datetime(2014, 1, 1, 0, 0, 0, 0, pytz.utc).date()
	#start = pd.Timestamp("2014-01-01", tz="utc")
	end = datetime.datetime(2012, 11, 1, 0, 0, 0, 0, datetime.timezone.utc).date()
	#end = datetime.datetime(2014, 11, 1, 0, 0, 0, 0, pytz.utc).date()
	#end = pd.Timestamp("2014-11-01", tz="utc")

	zipline.run_algorithm(
		start=start,
		end=end,
		initialize=initialize,
		capital_base=1e7,
		handle_data=handle_data,
		before_trading_start=None,
		analyze=analyze,
		data_frequency="daily",
		bundle="quantopian-quandl",  # zipline ingest -b quantopian-quandl
		bundle_timestamp=None,
		trading_calendar=None,
		metrics_set="default",
		benchmark_returns=None,
		default_extension=True,
		extensions=(),
		strict_extensions=True,
		environ=os.environ,
		blotter="default"
	)

def main():
	#quickstart_tutorial()  # Not working.
	buyapple_example()  # Not working.

#--------------------------------------------------------------------

# Installation:
#	conda install -c conda-forge zipline
#	pip install Zipline

# Usage:
#	Zipline CLI:
#		zipline ingest
#		zipline run -f dual_moving_average.py --start 2014-1-1 --end 2018-1-1 -o dma.pickle --no-benchmark

if "__main__" == __name__:
	main()
