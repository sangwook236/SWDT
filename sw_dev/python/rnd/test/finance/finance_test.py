#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://mokeya.tistory.com/149
def open_dart_api_test():
	raise NotImplementedError

# REF [site] >> https://github.com/josw123/dart-fss
def dart_fss_test():
	import os
	import dart_fss as dart

	api_key = os.getenv("OPEN_DART_API_KEY", "")
	if api_key == "":
		print("Error: OPEN_DART_API_KEY is not set.")
		return
	dart.set_api_key(api_key=api_key)

	#-----
	# 기업정보검색
	# https://dart-fss.readthedocs.io/en/latest/dart_corp.html

	# DART 에 공시된 회사 리스트 불러오기
	corp_list = dart.get_corp_list()
	print(corp_list)
	print(f"{len(corp_list)=}.")
	print(f"{corp_list.sectors=}.")

	#-----
	# dart_fss.corp.CorpList

	# 삼성전자 검색
	samsung = corp_list.find_by_corp_name("삼성전자", exactly=True)[0]
	#corps = corp_list.find_by_corp_name("삼성")
	#samsung = corp_list.find_by_corp_code("00126380")[0]  # 회사 코드.
	#samsung = corp_list.find_by_stock_code("005930", include_delisting=False, include_trading_halt=True)  # 증권 코드.
	#corps = corp_list.find_by_product("휴대폰", market='YKN')  # 상품. 'Y': 코스피, 'K': 코스닥, 'N': 코넥스, 'E': 기타.
	#corps = corp_list.find_by_sector("텔레비전 방송업", market='YKN')  # 산업 섹터. 'Y': 코스피, 'K': 코스닥, 'N': 코넥스, 'E': 기타.
	#corp_list.load(profile=False)

	#-----
	# dart_fss.corp.Corp

	samsung = corp_list.find_by_corp_name("삼성전자", exactly=True)[0]

	print(f"{samsung.corp_code=}.")  # 종목 코드
	print(f"{samsung.corp_name=}.")  # 종목 이름
	print(f"{samsung.stock_code=}.")  # 주식 종목 코드
	print(f"{samsung.modify_date=}.")  # 최종 업데이트 일자

	print(f"{samsung.info=}.")
	print(f"{samsung.to_dict()=}.")

	if False:
		# 공시보고서 검색
		#	https://dart-fss.readthedocs.io/en/latest/dart_search.html

		# dart_fss.corp.Corp.search_filings(bgn_de=None, end_de=None, last_reprt_at='N', pblntf_ty=None, pblntf_detail_ty=None, corp_cls=None, sort='date', sort_mth='desc', page_no=1, page_count=10)
		# dart_fss.filings.search(corp_code=None, bgn_de=None, end_de=None, last_reprt_at='N', pblntf_ty=None, pblntf_detail_ty=None, corp_cls=None, sort='date', sort_mth='desc', page_no=1, page_count=10)

		# 2019년 3월 1일부터 2019년 5월 31일까지 삼성전자의 모든 공시 정보 조회
		reports = samsung.search_filings(bgn_de="20190301", end_de="20190531")
		print(f"{reports=}.")

		# 2010년 1월 1일부터 현재까지 모든 사업보고서 검색
		reports = samsung.search_filings(bgn_de="20100101", pblntf_detail_ty="a001")
		print(f"{reports=}.")

		# 2010년 1월 1일부터 현재까지 모든 사업보고서의 최종보고서만 검색
		reports = samsung.search_filings(bgn_de="20100101", pblntf_detail_ty="a001", last_reprt_at="Y")
		print(f"{reports=}.")

		# 2010년 1월 1일부터 현재까지 사업보고서, 반기보고서, 분기보고서 검색
		reports = samsung.search_filings(bgn_de="20100101", pblntf_detail_ty=["a001", "a002", "a003"])
		print(f"{reports=}.")

	if False:
		# 재무제표 검색
		#	https://dart-fss.readthedocs.io/en/latest/dart_fs.html

		# dart_fss.corp.Corp.extract_fs(bgn_de, end_de=None, fs_tp=('bs', 'is', 'cis', 'cf'), separate=False, report_tp='annual', lang='ko', separator=True, dataset='xbrl', cumulative=False, progressbar=True, skip_error=True, last_report_only=True)
		# dart_fss.fs.extract(corp_code, bgn_de, end_de=None, fs_tp=('bs', 'is', 'cis', 'cf'), separate=False, report_tp='annual', lang='ko', separator=True, dataset='xbrl', cumulative=False, progressbar=True, skip_error=True, last_report_only=True)

		# 2012년 1월 1일부터 현재까지 연간 연결재무제표 검색
		fs = samsung.extract_fs(bgn_de="20120101")

		# 재무제표 검색 결과를 엑셀파일로 저장 (기본저장위치: 실행폴더/fsdata)
		#fs.save()

		# 2012년 1월 1일부터 현재까지 분기 연결재무제표 검색 (연간보고서, 반기보고서 포함)
		fs_quarter = samsung.extract_fs(bgn_de="20120101", report_tp="quarter")

		# 2012년 1월 1일부터 현재까지 개별재무제표 검색
		fs_separate = samsung.extract_fs(bgn_de="20120101", separate=True)

# REF [site] >> https://github.com/FinanceData/OpenDartReader
def OpenDartReader_test():
	raise NotImplementedError

# REF [site] >>
def kind_test():
	import pandas as pd

	df = pd.read_html("https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13", encoding="euc-kr")[0]
	print(df.head())

	df["종목코드"] = df["종목코드"].map("{:06d}".format)
	df = df.sort_values(by="종목코드")
	print(df.head())

# REF [site] >> https://mokeya.tistory.com/70
def company_guide_test():
	raise NotImplementedError

# REF [site] >> https://github.com/mortada/fredapi
def fredapi_test():
	raise NotImplementedError

# REF [site] >> https://github.com/ranaroussi/yfinance
def yfinance_test():
	import yfinance as yf

	#-----
	# Ticker module

	"""
	yfinance.Ticker members:
		actions, analyst_price_target, balance_sheet, balancesheet, basic_info, calendar, capital_gains, cash_flow, cashflow,
		dividends, earnings, earnings_dates, earnings_forecasts, earnings_trend, fast_info, financials,
		get_actions, get_analyst_price_target, get_balance_sheet, get_balancesheet, get_calendar, get_capital_gains,
		get_cash_flow, get_cashflow, get_dividends, get_earnings, get_earnings_dates, get_earnings_forecast, get_earnings_trend,
		get_fast_info, get_financials, get_history_metadata, get_income_stmt, get_incomestmt, get_info, get_institutional_holders,
		get_isin, get_major_holders, get_mutualfund_holders, get_news, get_recommendations, get_recommendations_summary, 
		get_rev_forecast, get_shares, get_shares_full, get_splits, get_sustainability, get_trend_details, history, history_metadata,
		income_stmt, incomestmt, info, institutional_holders, isin, major_holders, mutualfund_holders, news, option_chain, options,
		quarterly_balance_sheet, quarterly_balancesheet, quarterly_cash_flow, quarterly_cashflow, quarterly_earnings,
		quarterly_financials, quarterly_income_stmt, quarterly_incomestmt,
		recommendations, recommendations_summary, revenue_forecasts, session, shares, splits, sustainability, ticker, trend_details
	"""

	#ticker = yf.Ticker(ticker, session=None)
	ticker = yf.Ticker("MSFT")
	#ticker = yf.Ticker("AAPL")
	#ticker = yf.Ticker("GOOG")

	# Get all stock info
	print(f"{ticker.info=}.")
	#ticker.get_info(proxy=None)
	print(f"{ticker.basic_info=}.")
	print(f"{ticker.fast_info=}.")
	#ticker.get_fast_info(proxy=None)
	print(f"{ticker.session=}.")
	print(f"{ticker.ticker=}.")

	# Get historical market data
	#hist = ticker.history(period="1mo", interval="1d", start=None, end=None, prepost=False, actions=True, auto_adjust=True, back_adjust=False, repair=False, keepna=False, proxy=None, rounding=False, timeout=10, debug=None, raise_errors=False)
	hist = ticker.history(period="1mo")

	print("History: ----------")
	print(hist)

	# Show meta information about the history (requires history() to be called first)
	print(f"{ticker.history_metadata=}.")
	#ticker.get_history_metadata(proxy=None)

	# Show actions (dividends, splits, capital gains)
	print("Actions: ----------")
	print(ticker.actions)
	#ticker.get_actions(proxy=None)
	print("Dividends: ----------")
	print(ticker.dividends)
	#ticker.get_dividends(proxy=None)
	print("Splits: ----------")
	print(ticker.splits)
	#ticker.get_splits(proxy=None)
	print("Capital gains: ----------")
	print(ticker.capital_gains)  # Only for mutual funds & etfs
	#ticker.get_capital_gains(proxy=None)

	# Show share count
	print("Share count: ----------")
	print(ticker.get_shares_full(start="2022-01-01", end=None))

	# Show financials (financial statements)
	# - Income statement
	print("Income statement: ----------")
	print(ticker.income_stmt)
	#ticker.incomestmt
	#ticker.get_income_stmt(proxy=None, as_dict=False, pretty=False, freq="yearly")
	#ticker.get_incomestmt(proxy=None, as_dict=False, pretty=False, freq="yearly")
	#ticker.financials
	#ticker.get_financials(proxy=None, as_dict=False, pretty=False, freq="yearly")
	print("Quarterly income statement: ----------")
	print(ticker.quarterly_income_stmt)
	#ticker.quarterly_incomestmt
	#ticker.quarterly_financials
	# - Balance sheet
	print("Balance sheet: ----------")
	print(ticker.balance_sheet)
	#ticker.balancesheet
	ticker.get_balance_sheet(proxy=None, as_dict=False, pretty=False, freq="yearly")
	#ticker.get_balancesheet(proxy=None, as_dict=False, pretty=False, freq="yearly")
	print("Quarterly balance sheet: ----------")
	print(ticker.quarterly_balance_sheet)
	#ticker.quarterly_balancesheet
	# - Cash flow statement
	print("Cash flow statement: ----------")
	print(ticker.cash_flow)
	#ticker.cashflow
	#ticker.get_cash_flow(proxy=None, as_dict=False, pretty=False, freq="yearly")
	#ticker.get_cashflow(proxy=None, as_dict=False, pretty=False, freq="yearly")
	print("Quarterly cash flow statement: ----------")
	print(ticker.quarterly_cash_flow)
	#ticker.quarterly_cashflow

	# Show holders
	print("Major holders: ----------")
	print(ticker.major_holders)
	#ticker.get_major_holders(proxy=None, as_dict=False)
	print("Institutional holders: ----------")
	print(ticker.institutional_holders)
	#ticker.get_institutional_holders(proxy=None, as_dict=False)
	print("Mutual fund holders: ----------")
	print(ticker.mutualfund_holders)
	#ticker.get_mutualfund_holders(proxy=None, as_dict=False)

	# Show future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default. 
	# Note: If more are needed use ticker.get_earnings_dates(limit=XX) with increased limit argument.
	print("Earnings dates: ----------")
	print(ticker.earnings_dates)
	ticker.get_earnings_dates(limit=12, proxy=None)

	# Show ISIN code - *experimental*
	# ISIN = International Securities Identification Number
	print(f"{ticker.isin=}.")
	#ticker.get_isin(proxy=None)

	# Show options expirations
	print("Options expirations: ----------")
	print(ticker.options)

	# Get option chain for specific expiration
	#opt = ticker.option_chain(date=None, proxy=None, tz=None)  # yfinance.ticker.Options
	opt = ticker.option_chain("2024-02-16")
	# Data available via: opt.calls, opt.puts

	print("Option chain for specific expiration: ----------")
	print(opt)

	# Show news
	print("News: ----------")
	print(ticker.news)
	#ticker.get_news(proxy=None)

	try:
		ticker.earnings
		ticker.get_earnings(proxy=None, as_dict=False, freq="yearly")
		ticker.earnings_trend
		ticker.get_earnings_trend(proxy=None, as_dict=False)
		ticker.earnings_forecasts
		ticker.get_earnings_forecast(proxy=None, as_dict=False)
		ticker.quarterly_earnings
		ticker.shares
		ticker.get_shares(proxy=None, as_dict=False)
		ticker.calendar
		ticker.get_calendar(proxy=None, as_dict=False)
		ticker.analyst_price_target
		ticker.get_analyst_price_target(proxy=None, as_dict=False)
		ticker.recommendations
		ticker.get_recommendations(proxy=None, as_dict=False)
		ticker.recommendations_summary
		ticker.get_recommendations_summary(proxy=None, as_dict=False)
		ticker.revenue_forecasts
		ticker.get_rev_forecast(proxy=None, as_dict=False)
		ticker.sustainability
		ticker.get_sustainability(proxy=None, as_dict=False)
		ticker.trend_details
		ticker.get_trend_details(proxy=None, as_dict=False)
	except yf.exceptions.YFNotImplementedError as ex:
		print(f"YFNotImplementedError raised: {ex}.")

	#-----
	# Multiple tickers

	#tickers = yf.Tickers(tickers, session=None)
	tickers = yf.Tickers("msft aapl goog")

	# Access each ticker using (example)
	tickers.tickers["MSFT"].info
	tickers.tickers["AAPL"].history(period="1mo")
	tickers.tickers["GOOG"].actions

	#-----
	# Download price history into one table

	"""
	df = yf.download(
		tickers, start=None, end=None, actions=False, threads=True, ignore_tz=None,
		group_by="column", auto_adjust=False, back_adjust=False, repair=False, keepna=False,
		progress=True, period="max", show_errors=None, interval="1d", prepost=False,
		proxy=None, rounding=False, timeout=10, session=None
	)
	"""
	data_df = yf.download("SPY AAPL", period="1mo")
	print(data_df.head())

# REF [site] >> https://github.com/WooilJeong/PublicDataReader
def PublicDataReader_test():
	raise NotImplementedError

# REF [site] >> https://github.com/FinanceData/FinanceDataReader
def FinanceDataReader_test():
	raise NotImplementedError

# REF [site] >> http://pmorissette.github.io/bt/
def bt_quick_example():
	import bt

	# Fetch some data
	data = bt.get('spy,agg', start='2010-01-01')
	print(data.head())

	# Create the strategy
	s = bt.Strategy('s1', [
		bt.algos.RunMonthly(),
		bt.algos.SelectAll(),
		bt.algos.WeighEqually(),
		bt.algos.Rebalance()
	])

	# Create a backtest and run it
	test = bt.Backtest(s, data)
	res = bt.run(test)

	# Frst let's see an equity curve
	res.plot();

	# What about some stats?
	res.display()

	# How does the return distribution look like?
	res.plot_histogram()

	# Just to make sure everything went along as planned, let's plot the security weights over time
	res.plot_security_weights()

	#-----
	# Modifying a strategy

	# Create our new strategy
	s2 = bt.Strategy('s2', [
		bt.algos.RunWeekly(),
		bt.algos.SelectAll(),
		bt.algos.WeighInvVol(),
		bt.algos.Rebalance()
	])

	# Now let's test it with the same data set. We will also compare it with our first backtest.
	test2 = bt.Backtest(s2, data)
	# We include test here to see the results side-by-side
	res2 = bt.run(test, test2)

	res2.plot()

# REF [site] >> http://pmorissette.github.io/bt/examples.html
def bt_examples():
	raise NotImplementedError

# REF [site] >> https://github.com/polakowo/vectorbt
def vectorbt_test():
	raise NotImplementedError

# REF [site] >> https://github.com/twopirllc/pandas-ta
def pandas_ta_test():
	raise NotImplementedError

def main():
	# Financial data

	# KRX
	#	Refer to ./pykrx_test
	#	Refer to ./krx_test

	# DART (FSS)
	#open_dart_api_test()  # Not yet implemented
	#dart_fss_test()
	#OpenDartReader_test()  # Not yet implemented

	# KIND (KRX)
	#kind_test()

	# CompanyGuide (FnGuide)
	#company_guide_test()  # Not yet implemented

	# FRED
	#fredapi_test()  # Not yet implemented

	yfinance_test()
	#PublicDataReader_test()  # Not yet implemented
	#FinanceDataReader_test()  # Not yet implemented

	#-----
	# Stock statistics and indicators

	# stockstats
	#	Refer to ./stockstats_test.py

	#-----
	# Backtesting & forward testing

	#bt_quick_example()  # Not yet tested
	#bt_examples()  # Not yet implemented
	#vectorbt_test()  # Not yet implemented

	#-----
	# Technical analysis

	# TA-Lib
	#	Refer to ./ta_lib_test.py

	#pandas_ta_test()  # Not yet implemented

	#-----
	# Algorithmic trading

	# Zipline
	#	Refer to ./zipline_test.py
	# FinRL, FinRL-Meta, FinRL-Trading, FinGPT, FinNLP
	#	Refer to ./finrl_test.py
	# Refer to ./trading_bot.py

	#-----
	# Quantitative investment

	# Refer to ./quant_test.py

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
