#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import io
from bs4 import BeautifulSoup
import imgkit
import PIL.Image, PIL.ImageFont
import matplotlib.pyplot as plt

# REF [site] >> https://blog.finxter.com/how-to-parse-html-table-using-python/
def table_test():
	table_tags1 = """
<table>
<thead>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td colspan=6></td></tr>
</thead>
<tbody>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>
"""
	table_tags2 = """
<table>
<thead>
<tr><td></td><td></td><td></td><td colspan=2></td></tr>
</thead>
<tbody>
<tr><td></td><td></td><td></td><td rowspan=2></td><td></td></tr>
<tr><td></td><td></td><td></td><td></td></tr>
<tr><td></td><td></td><td></td><td rowspan=2></td><td></td></tr>
<tr><td></td><td></td><td></td><td></td></tr>
</tbody>
</table>
"""

	soup = BeautifulSoup(table_tags2, features="lxml")
	#soup = BeautifulSoup(table_tags2, features="html5lib")

	table = soup.find("table")

	if table.thead:
		print("Table header --------------------")
		rows = table.thead.find_all("tr")
		#print(rows)
		for row in rows:
			cols = row.find_all("th")
			print(cols)
			cols = row.find_all("td")
			print(cols)

			#td_spans = row.find_all("td", {"rowspan": "2"})
			td_spans = row.find_all(lambda tag: tag.name == "td" and ("rowspan" in tag.attrs.keys() or "colspan" in tag.attrs.keys()))
			for cell in td_spans:
				print("\t{}: rowspan = {}, colspan = {}.".format(cell.name, cell.attrs["rowspan"] if "rowspan" in cell.attrs else 1, cell.attrs["colspan"] if "colspan" in cell.attrs else 1))

	if table.tbody:
		print("Table body --------------------")
		rows = table.tbody.find_all("tr")
		#print(rows)
		for row in rows:
			cols = row.find_all("td")
			print(cols)

			#td_spans = row.find_all("td", {"rowspan": "2"})
			td_spans = row.find_all(lambda tag: tag.name == "td" and ("rowspan" in tag.attrs.keys() or "colspan" in tag.attrs.keys()))
			for cell in td_spans:
				print("\t{}: rowspan = {}, colspan = {}.".format(cell.name, cell.attrs["rowspan"] if "rowspan" in cell.attrs else 1, cell.attrs["colspan"] if "colspan" in cell.attrs else 1))

	if table.tfoot:
		print("Table footer --------------------")
		rows = table.tfoot.find_all("tr")
		#print(rows)
		for row in rows:
			cols = row.find_all("td")
			print(cols)

			#td_spans = row.find_all("td", {"rowspan": "2"})
			td_spans = row.find_all(lambda tag: tag.name == "td" and ("rowspan" in tag.attrs.keys() or "colspan" in tag.attrs.keys()))
			for cell in td_spans:
				print("\t{}: rowspan = {}, colspan = {}.".format(cell.name, cell.attrs["rowspan"] if "rowspan" in cell.attrs else 1, cell.attrs["colspan"] if "colspan" in cell.attrs else 1))

def construct_html_table_page(table_tags):
	html_page = """<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head>
    <title>PubTabNet HTML Table!</title>
    <style>
    table, th, td {{
      border: 1px solid black;
      border-collapse: collapse;
    }}
    table {{
      border-spacing: 2px;
    }}
    th, td {{
      padding: 5px;
    }}
    th {{
       text-align: center;
    }}
    </style>
  </head>
  <body>
    <div id="header">
      <h2>PubTabNet HTML Table!</h2>
    </div>

    <table style="width:100%">
      {table}
    </table>

    <div id="footer">
      <hr />
      <p>https://github.com/sangwook236/SWDT</p>
    </div>
  </body>
</html>
"""
	return html_page.format(table=table_tags)

def visualize_table():
	table_tags = """
<thead>
<tr><td>A</td><td>B</td><td>C</td><td colspan=2>D</td></tr>
</thead>
<tbody>
<tr><td>11</td><td>12</td><td>13</td><td rowspan=2>14</td><td>15</td></tr>
<tr><td>21</td><td>22</td><td>23</td><td>24</td></tr>
<tr><td>31</td><td>32</td><td>33</td><td rowspan=2>34</td><td>35</td></tr>
<tr><td>41</td><td>42</td><td>43</td><td>44</td></tr>
</tbody>
"""

	#html_page = BeautifulSoup("<table>" + table_tags + "</table>", features="lxml").prettify()
	html_page = construct_html_table_page("".join(table_tags))
	#print(html_page)

	# Show a HTML.
	html_img = imgkit.from_string(html_page, False)
	html_img = PIL.Image.open(io.BytesIO(html_img))
	#plt.figure()
	plt.imshow(html_img)
	plt.tight_layout()
	#plt.title("HTML Table Image")
	plt.axis("off")
	plt.show()

def main():
	table_test()
	#visualize_table()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
