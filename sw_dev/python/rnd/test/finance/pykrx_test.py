#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
import pykrx.stock
import pykrx.website.krx.bond
import pandas as pd

# REF [site] >> https://github.com/sharebook-kr/pykrx
def market_data_example():
	if False:
		#tickers = pykrx.stock.get_market_ticker_list()
		#tickers = pykrx.stock.get_market_ticker_list("20190225")
		tickers = pykrx.stock.get_market_ticker_list("20190225", market="KOSDAQ")
		print(tickers)

		for ticker in pykrx.stock.get_market_ticker_list():
			ticker_name = pykrx.stock.get_market_ticker_name(ticker)
			print(ticker_name)

	if False:
		df = pykrx.stock.get_market_ohlcv_by_date("20150720", "20150810", "005930")
		#df = pykrx.stock.get_market_ohlcv_by_date("20180810", "20181212", "005930", freq="m")
		print(df.head(3))

		for ticker in pykrx.stock.get_market_ticker_list():
			df = pykrx.stock.get_market_ohlcv_by_date("20181210", "20181212", ticker)
			print(df.head())
			time.sleep(1)

	#--------------------
	if False:
		df = pykrx.stock.get_market_ohlcv_by_ticker("20210122")
		#df = pykrx.stock.get_market_ohlcv_by_ticker("20200831", market="KOSPI")
		#df = pykrx.stock.get_market_ohlcv_by_ticker("20200831", market="KOSDAQ")
		#df = pykrx.stock.get_market_ohlcv_by_ticker("20200831", market="KONEX")
		print(df.head(3))

		df = pykrx.stock.get_market_price_change_by_ticker("20180301", "20180320")
		print(df.head(2))

	#--------------------
	# DIV/BPS/PER/EPS.
	if False:
		df = pykrx.stock.get_market_fundamental_by_ticker("20210108")
		#df = pykrx.stock.get_market_fundamental_by_ticker("20210104", market="KOSDAQ")
		print(df.head(2))

		df = pykrx.stock.get_market_fundamental_by_date("20210104", "20210108", "005930")
		#df = pykrx.stock.get_market_fundamental_by_date("20200101", "20200430", "005930", freq="m")
		print(df.head(2))

	#--------------------
	if False:
		df = pykrx.stock.get_market_trading_value_by_date("20210115", "20210122", "005930")
		#df = pykrx.stock.get_market_trading_value_by_date("20210115", "20210122", "005930", on="매도")
		#df = pykrx.stock.get_market_trading_value_by_date("20210115", "20210122", "KOSPI")
		#df = pykrx.stock.get_market_trading_value_by_date("20210115", "20210122", "KOSPI", etf=True, etn=True, elw=True)
		#df = pykrx.stock.get_market_trading_value_by_date("20210115", "20210122", "KOSPI", etf=True, etn=True, elw=True, detail=True)
		print(df.head(2))

	#--------------------
	if False:
		df = pykrx.stock.get_market_trading_volume_by_date("20210115", "20210122", "005930")
		#df = pykrx.stock.get_market_trading_volume_by_date("20210115", "20210122", "005930", on="매도")
		#df = pykrx.stock.get_market_trading_volume_by_date("20210115", "20210122", "KOSPI")
		#df = pykrx.stock.get_market_trading_volume_by_date("20210115", "20210122", "KOSPI", etf=True, etn=True, elw=True)
		#df = pykrx.stock.get_market_trading_volume_by_date("20210115", "20210122", "KOSPI", etf=True, etn=True, elw=True, detail=True)
		print(df.head())

	#--------------------
	if False:
		df = pykrx.stock.get_market_trading_value_by_investor("20210115", "20210122", "005930")
		#df = pykrx.stock.get_market_trading_value_by_investor("20210115", "20210122", "KOSPI")
		#df = pykrx.stock.get_market_trading_value_by_investor("20210115", "20210122", "KOSPI", etf=True, etn=True, elw=True)
		print(df.head())

	#--------------------
	if False:
		df = pykrx.stock.get_market_trading_volume_by_investor("20210115", "20210122", "005930")
		#df = pykrx.stock.get_market_trading_volume_by_investor("20210115", "20210122", "KOSPI")
		#df = pykrx.stock.get_market_trading_volume_by_investor("20210115", "20210122", "KOSPI", etf=True, etn=True, elw=True)
		print(df.head())

	#--------------------
	if False:
		df = pykrx.stock.get_market_net_purchases_of_equities_by_ticker("20210115", "20210122", "KOSPI", "개인")
		print(df.head())

	#--------------------
	if False:
		df = pykrx.stock.get_market_cap_by_ticker("20200625")
		print(df.head())

		df = pykrx.stock.get_market_cap_by_date("20190101", "20190131", "005930")
		#df = pykrx.stock.get_market_cap_by_date("20200101", "20200430", "005930", freq="m")
		print(df.head())

	#--------------------
	if False:
		df = pykrx.stock.get_exhaustion_rates_of_foreign_investment_by_ticker("20200703")
		#df = pykrx.stock.get_exhaustion_rates_of_foreign_investment_by_ticker("20200703", "KOSPI")
		#df = pykrx.stock.get_exhaustion_rates_of_foreign_investment_by_ticker("20200703", "KOSPI", balance_limit=True)
		print(df.head())

		df = pykrx.stock.get_exhaustion_rates_of_foreign_investment_by_date("20210108", "20210115", "005930")
		print(df.head())

# REF [site] >> https://github.com/sharebook-kr/pykrx
def index_example():
	tickers = pykrx.stock.get_index_ticker_list()
	tickers = pykrx.stock.get_index_ticker_list("19800104")
	tickers = pykrx.stock.get_index_ticker_list(market="KOSDAQ")
	print(tickers)

	for ticker in pykrx.stock.get_index_ticker_list():
		print(ticker, pykrx.stock.get_index_ticker_name(ticker))

	pdf = pykrx.stock.get_index_portfolio_deposit_file("1005")
	print(len(pdf), pdf)

	df = pykrx.stock.get_index_ohlcv_by_date("20190101", "20190228", "1028")
	#df = pykrx.stock.get_index_ohlcv_by_date("20190101", "20190228", "1028", freq="m")
	print(df.head(2))

	df = pykrx.stock.get_index_listing_date("KOSPI")
	print(df.head())

	df = pykrx.stock.get_index_price_change_by_ticker("20200520", "20200527", "KOSDAQ")
	print(df.head())

# REF [site] >> https://github.com/sharebook-kr/pykrx
def short_stock_selling_example():
	df = pykrx.stock.get_shorting_status_by_date("20181210", "20181212", "005930")
	print(df)

	df = pykrx.stock.get_shorting_volume_by_ticker("20210125")
	#df = pykrx.stock.get_shorting_volume_by_ticker("20210125", "KOSDAQ")
	#df = pykrx.stock.get_shorting_volume_by_ticker("20210125", include=["주식", "ELW"])
	print(df.head())

	df = pykrx.stock.get_shorting_volume_by_date("20210104", "20210108", "005930")
	print(df.head(3))

	df = pykrx.stock.get_shorting_investor_volume_by_date("20190401", "20190405", "KOSPI")
	#df = pykrx.stock.get_shorting_investor_volume_by_date("20190401", "20190405", "KOSDAQ")
	print(df.head())

	df = pykrx.stock.get_shorting_investor_value_by_date("20190401", "20190405", "KOSPI")
	#df = pykrx.stock.get_shorting_investor_value_by_date("20190401", "20190405", "KOSDAQ")
	print(df.head())

	df = pykrx.stock.get_shorting_balance_by_date("20190401", "20190405", "005930")
	print(df.head())

	df = pykrx.stock.get_shorting_volume_top50("20210129")
	#df = pykrx.stock.get_shorting_volume_top50("20210129", "KOSDAQ")
	print(df.head())

	df = pykrx.stock.get_shorting_balance_top50("20210127")
	#df = pykrx.stock.get_shorting_balance_top50("20210129", market="KOSDAQ")
	print(df.head())

# REF [site] >> https://github.com/sharebook-kr/pykrx
def etx_example():
	#--------------------
	# ETF.
	tickers = pykrx.stock.get_etf_ticker_list("20200717")
	print(tickers[:10])

	tickers = pykrx.stock.get_etf_ticker_list("20021014")
	for ticker in tickers:
		print(pykrx.stock.get_etf_ticker_name(ticker))

	df = pykrx.stock.get_etf_ohlcv_by_date("20210104", "20210108", "292340")
	df = pykrx.stock.get_etf_ohlcv_by_date("20200101", "20200531", "292340", freq="m")
	print(df.head())

	df = pykrx.stock.get_etf_ohlcv_by_ticker("20210325")
	print(df.head())

	df = pykrx.stock.get_etf_price_change_by_ticker("20210325", "20210402")
	print(df.head())

	df = pykrx.stock.get_etf_portfolio_deposit_file("152100")
	#df = pykrx.stock.get_etf_portfolio_deposit_file("152100", "20161206")
	print(df.head())

	df = pykrx.stock.get_etf_price_deviation("20200101", "20200401", "295820")
	print(df.head())

	df = pykrx.stock.get_etf_tracking_error("20210104", "20210108", "295820")
	print(df.head())

	#--------------------
	# ETN.
	tickers = pykrx.stock.get_etn_ticker_list("20141215")
	print(tickers)

	for ticker in tickers:
		print(pykrx.stock.get_etn_ticker_name(ticker))

	#--------------------
	# ELW.
	tickers = pykrx.stock.get_elw_ticker_list("20200306")
	print(tickers)

	for ticker in tickers:
		print(pykrx.stock.get_elw_ticker_name(ticker))

# REF [site] >> https://github.com/sharebook-kr/pykrx
def bond_example():
	kb = pykrx.website.krx.bond.KrxBond()
	df = kb.get_treasury_yields_in_kerb_market("20190208")
	print(df)

def main():
	market_data_example()
	#index_example()
	#short_stock_selling_example()
	#etx_example()
	#bond_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
