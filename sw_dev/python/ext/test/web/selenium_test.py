#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from selenium import webdriver

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

def main():
	getting_started_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
