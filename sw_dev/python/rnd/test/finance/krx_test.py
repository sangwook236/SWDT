#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import io, urllib.request, urllib.error, time, datetime
import pandas as pd
import sqlite3
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# REF [site] >> https://tariat.tistory.com/892
def simple_example_1():
	"""
	한국거래소 XML 서비스 URL.

	1. 실시간시세(국문).
	http://asp1.krx.co.kr/servlet/krx.asp.XMLSise?code=단축종목코드
	2. 실시간시세(영문).
	http://asp1.krx.co.kr/servlet/krx.asp.XMLSiseEng?code=단축종목코드
	3. 공시정보(국,영문).
	http://asp1.krx.co.kr/servlet/krx.asp.DisList4MainServlet?code=단축코드&gubun=K (K:국문/E:영문)
	4. 재무종합(국문)
	http://asp1.krx.co.kr/servlet/krx.asp.XMLJemu?code=단축종목코드
	5. 재무종합(영문).
	http://asp1.krx.co.kr/servlet/krx.asp.XMLJemuEng?code=단축종목코드
	6. 재무종합2(국문).
	http://asp1.krx.co.kr/servlet/krx.asp.XMLJemu2?code=단축종목코드
	7. 재무종합3(국문).
	http://asp1.krx.co.kr/servlet/krx.asp.XMLJemu3?code=단축종목코드
	8. 텍스트.
	http://asp1.krx.co.kr/servlet/krx.asp.XMLText?code=단축종목코드
	"""

	def get_stock_from_krx(stock_code, try_cnt):
		try:
			url = "http://asp1.krx.co.kr/servlet/krx.asp.XMLSiseEng?code={}".format(stock_code)

			req = urllib.request.urlopen(url)
			result = req.read()
			xmlsoup = BeautifulSoup(result, "lxml-xml")
			stock = xmlsoup.find("TBL_StockInfo")

			stock_df = pd.DataFrame(stock.attrs, index=[0])
			stock_df = stock_df.applymap(lambda x: x.replace(",", ""))
			return stock_df
		except urllib.error.HTTPError as ex:
			print("urllib.error.HTTPError raised: {}.".format(ex))
			if try_cnt >= 3:
				return None
			else:
				return get_stock_from_krx(stock_code, try_cnt=try_cnt + 1)

	# Save to DB.
	con = sqlite3.connect("./krx.db")
	stock_codes = ["005930", "066570"]

	for sc in tqdm(stock_codes):
		stock_df = get_stock_from_krx(sc, 1)
		stock_df.to_sql(con=con, name="div_stock_sise", if_exists="append")
		time.sleep(0.5)

	con.close()

def get_daily_price(date):
	gen_otp_url = "http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx"
	gen_otp_data = {
		"name": "fileDown",
		"filetype": "csv",
		"market_gubun": "ALL",
		"url": "MKD/04/0404/04040200/mkd04040200_01",
		"indx_ind_cd": "",
		"sect_tp_cd": "ALL",
		"schdate": date,
		"pagePath": "/contents/MKD/04/0404/04040200/MKD04040200.jsp"
	}
	headers = {
		"sec-fetch-dest": "empty",
		"sec-fetch-mode": "cors",
		"sec-fetch-site": "same-origin",
		"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
		"x-requested-with": "XMLHttpRequest"
	}

	r = requests.get(gen_otp_url, headers=headers, params=gen_otp_data)
	
	code = r.text 
	down_url = "http://file.krx.co.kr/download.jspx"
	down_data = {
		"code": code,
	}
	headers = {
		"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
		"accept-encoding": "gzip, deflate, br",
		"accept-language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
		"cache-control": "max-age=0",
		"content-length": "417",
		"content-type": "application/x-www-form-urlencoded",
		"origin": "https://marketdata.krx.co.kr",
		"referer": "https://marketdata.krx.co.kr/",
		"sec-fetch-dest": "iframe",
		"sec-fetch-mode": "navigate",
		"sec-fetch-site": "same-site",
		"sec-fetch-user": "?1",
		"upgrade-insecure-requests": "1",
		"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36"
	}

	r = requests.post(down_url, data=down_data, headers=headers)

	r.encoding = "utf-8-sig"
	df = pd.read_csv(io.BytesIO(r.content), header=0, thousands=",")
	#print(df)
	return df

# REF [site] >> https://leesunkyu94.github.io/투자%20전략/divdend_stra/
def simple_example_2():
	for i in range(0, 5):
		date = (datetime.datetime.today() - datetime.timedelta(days=i)).strftime("%Y%m%d") 
		data_df = get_daily_price(date)
		print(i, date)
		if data_df.shape[0] != 0:
			data_df.to_csv("./krx_{}.csv".format(date), encoding="CP949", index=False)

def main():
	simple_example_1()  # Not working.
	simple_example_2()  # Not correctly working.


#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
