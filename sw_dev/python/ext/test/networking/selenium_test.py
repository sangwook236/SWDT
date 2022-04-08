#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import time
import pandas as pd
from urllib.parse import quote_plus
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, SessionNotCreatedException, TimeoutException

# REF [site] >> https://www.selenium.dev/documentation/webdriver/getting_started/
def getting_started_example():
	# Start the session.
	if True:
		#driver = webdriver.Chrome()
		options = ChromeOptions()
		driver = webdriver.Chrome(options=options)
	elif False:
		options = EdgeOptions()
		driver = webdriver.Edge(options=options)
	elif False:
		options = FirefoxOptions()
		driver = webdriver.Firefox(options=options)
	elif False:
		options = ChromeOptions()
		options.binary_location = "/path/to/opera/browser"
		driver = webdriver.Chrome(options=options)
	elif False:
		driver = webdriver.Safari()

	# Take action on browser.
	driver.get("http://www.google.com")
	#driver.get("http://selenium.dev")

	# Request browser information.
	print(driver.title)

	# Establish Waiting Strategy.
	driver.implicitly_wait(0.5)

	# Find an element.
	search_box = driver.find_element(By.NAME, "q")
	search_button = driver.find_element(By.NAME, "btnK")

	# Take action on element.
	search_box.send_keys("Selenium")
	search_button.click()

	# Request element information.
	print(driver.find_element(By.NAME, "q").get_attribute("value"))

	# End the session.
	driver.quit()

def simple_example():
	# REF [site] >> https://chromedriver.chromium.org/
	chromedriver_filepath = 'D:\\util_portable\\web\\chromedriver_win32\\chromedriver.exe'

	_input  = input(
'''-월--일, -월, 이번주, 이번주말 중 선택하여 입력해주세요.
(-은 숫자 입력, 이번년도만 가능): '''
	)

	#url = f'https://search.naver.com/search.naver?where=nexearch&sm=tab_etc&query={_input}%20연극%20공연'
	user_input = quote_plus(_input)  # Convert Hangeul text to percent encoding.
	url = f'https://search.naver.com/search.naver?where=nexearch&sm=tab_etc&query={user_input}%20%EC%97%B0%EA%B7%B9%20%EA%B3%B5%EC%97%B0'

	#--------------------
	try:
		options = webdriver.ChromeOptions()
		options.add_argument('headless')  # Headless chrome. No web browser.
		options.add_argument('disable-gpu')
		options.add_argument('lang=ko_KR')

		driver = webdriver.Chrome(chromedriver_filepath, options=options)
	except WebDriverException as ex:
		print('WebDriverException raised: {}.'.format(ex))
		return
	except SessionNotCreatedException as ex:
		print('SessionNotCreatedException raised: {}.'.format(ex))
		return

	driver.get(url)

	#--------------------
	theaters = list()
	try:
		# 3 seconds pause until the tag is checked.
		element = WebDriverWait(driver, 3).until(
			EC.presence_of_element_located((By.CLASS_NAME, 'list_title'))
		)

		pageNum = int(driver.find_element_by_class_name('_totalCount').text)
		count = 0
		for i in range(1, pageNum):
			theater_data = driver.find_elements_by_class_name('list_title')
			image_data = driver.find_elements_by_class_name('list_thumb')

			for thr in theater_data:
				theaters.append(thr.text.split('\n'))
			for img in image_data:  # Image crawling.
				count += 1
				img.screenshot(f'img/{count}.png')

			driver.find_element_by_xpath("//a[@class='btn_page_next _btnNext on']").click()
			time.sleep(2)  # 2 seconds pause to load webpage.
	except TimeoutException as ex:
		print('TimeoutException raised: {}.'.format(ex))
	finally:
		driver.quit()

	#--------------------
	if theaters:
		for i in range(len(theaters)):
			theaters[i].append(theaters[i][1].split('~')[0])
			theaters[i].append(theaters[i][1].split('~')[1])

		for i in range(len(theaters)):
			if theaters[i][4] == '오픈런':
				theaters[i][4] = '50.01.01.'
				theaters[i].append('True')
			else:
				theaters[i].append('False')

		theater_df = pd.DataFrame(theaters, columns=['연극명', '기간', '장소', '개막일', '폐막일', '오픈런'])
		theater_df.index = theater_df.index + 1
		theater_df['개막일'] = pd.to_datetime(theater_df['개막일'], format='%y.%m.%d.')
		theater_df['폐막일'] = pd.to_datetime(theater_df['폐막일'], format='%y.%m.%d.')

		#theater_df.to_csv(f'./theater_{_input}_df.csv', mode='w', encoding='utf-8-sig', header=True, index=True)
		from IPython.display import display, HTML
		display(theater_df)
		#print(HTML(theater_df.to_html()).data)
	else:
		print('No theater found.')

def main():
	getting_started_example()
	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
