#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import io, functools
import pandas as pd
from bs4 import BeautifulSoup
import imgkit
import PIL.Image, PIL.ImageFont
import matplotlib.pyplot as plt

# REF [site] >> https://blog.finxter.com/how-to-parse-html-table-using-python/
def table_test():
	table_tags = """
<table>
<thead>
<tr><td>A</td><td>B</td><td colspan=2>C</td><td>D</td></tr>
</thead>
<tbody>
<tr><td>10</td><td>11</td><td>12</td><td rowspan=2>13</td><td>14</td></tr>
<tr><td rowspan=2 colspan=3>20</td><td>21</td></tr>
<tr><td rowspan=2>30</td><td>31</td></tr>
<tr><td>40</td><td>41</td><td>42</td><td>43</td></tr>
<tr><td>50</td><td>51</td><td>52</td><td>53</td><td>54</td></tr>
</tbody>
</table>
"""

	soup = BeautifulSoup(table_tags, features="lxml")
	#soup = BeautifulSoup(table_tags, features="html5lib")

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

			#spans = row.find_all("td", {"rowspan": "2"})
			spans = row.find_all(lambda tag: tag.name in ("th", "td") and ("rowspan" in tag.attrs.keys() or "colspan" in tag.attrs.keys()))
			for elem in spans:
				print("\t{}: rowspan = {}, colspan = {}.".format(elem.name, int(elem.attrs["rowspan"]) if "rowspan" in elem.attrs else 1, int(elem.attrs["colspan"]) if "colspan" in elem.attrs else 1))

	if table.tbody:
		print("Table body --------------------")
		rows = table.tbody.find_all("tr")
		#print(rows)
		for row in rows:
			cols = row.find_all("td")
			print(cols)

			#spans = row.find_all("td", {"rowspan": "2"})
			spans = row.find_all(lambda tag: tag.name in ("th", "td") and ("rowspan" in tag.attrs.keys() or "colspan" in tag.attrs.keys()))
			for elem in spans:
				print("\t{}: rowspan = {}, colspan = {}.".format(elem.name, int(elem.attrs["rowspan"]) if "rowspan" in elem.attrs else 1, int(elem.attrs["colspan"]) if "colspan" in elem.attrs else 1))

	if table.tfoot:
		print("Table footer --------------------")
		rows = table.tfoot.find_all("tr")
		#print(rows)
		for row in rows:
			cols = row.find_all("td")
			print(cols)

			#spans = row.find_all("td", {"rowspan": "2"})
			spans = row.find_all(lambda tag: tag.name in ("th", "td") and ("rowspan" in tag.attrs.keys() or "colspan" in tag.attrs.keys()))
			for elem in spans:
				print("\t{}: rowspan = {}, colspan = {}.".format(elem.name, int(elem.attrs["rowspan"]) if "rowspan" in elem.attrs else 1, int(elem.attrs["colspan"]) if "colspan" in elem.attrs else 1))

def index_table_cells():
	table_tags = """
<table>
<thead>
<tr><td>A</td><td>B</td><td colspan=2>C</td><td>D</td></tr>
</thead>
<tbody>
<tr><td>10</td><td>11</td><td>12</td><td rowspan=2>13</td><td>14</td></tr>
<tr><td rowspan=2 colspan=3>20</td><td>21</td></tr>
<tr><td rowspan=2>30</td><td>31</td></tr>
<tr><td>40</td><td>41</td><td>42</td><td>43</td></tr>
<tr><td>50</td><td>51</td><td>52</td><td>53</td><td>54</td></tr>
</tbody>
</table>
"""

	soup = BeautifulSoup(table_tags, features="lxml")
	#soup = BeautifulSoup(table_tags, features="html5lib")

	table = soup.find("table")

	#--------------------
	def construct_grid_table(table):
		grid_table = dict()
		row_idx, max_col_idx = 0, 0
		for row in table.find_all("tr"):
			col_idx = 0
			for col in row.find_all(lambda tag: tag.name in ("th", "td")):
				while (row_idx, col_idx) in grid_table:
					col_idx += 1
				row_span = max(int(col.attrs["rowspan"]) if "rowspan" in col.attrs else 1, 1)
				col_span = max(int(col.attrs["colspan"]) if "colspan" in col.attrs else 1, 1)
				for _ in range(col_span):
					for ri in range(row_span):
						grid_table[(row_idx + ri, col_idx)] = col.text
					col_idx += 1
			row_idx += 1
			max_col_idx = max(max_col_idx, col_idx)
		return grid_table, (row_idx, max_col_idx)

	grid_table, table_size = construct_grid_table(table)
	#if table.thead:
	#	grid_table, table_size = construct_grid_table(table.thead)
	#if table.tbody:
	#	grid_table, table_size = construct_grid_table(table.tbody)
	#if table.tfoot:
	#	grid_table, table_size = construct_grid_table(table.tfoot)

	print("Grid table = {}.".format(grid_table))
	print("Grid table size = {}.".format(table_size))

	#-----
	#num_rows = functools.reduce(lambda maxval, rc: max(maxval, rc[0]), grid_table, 0) + 1
	#num_cols = functools.reduce(lambda maxval, rc: max(maxval, rc[1]), grid_table, 0) + 1
	num_rows, num_cols = table_size

	table_data = [[None] * num_cols for _ in range(num_rows)]
	for (row_idx, col_idx), cell_text in grid_table.items():
		table_data[row_idx][col_idx] = cell_text
	#print(table_data)

	table_df = pd.DataFrame(table_data) 
	print(table_df)

	#--------------------
	def get_table_cell_start_indices(table_component, start_row_idx=0):
		cell_start_indices = list()
		grid_table = dict()
		row_idx = 0
		for row in table_component.find_all("tr"):
			col_idx = 0
			for col in row.find_all(lambda tag: tag.name in ("th", "td")):
				while (row_idx, col_idx) in grid_table:
					col_idx += 1
				cell_start_indices.append((start_row_idx + row_idx, col_idx))
				row_span = max(int(col.attrs["rowspan"]) if "rowspan" in col.attrs else 1, 1)
				col_span = max(int(col.attrs["colspan"]) if "colspan" in col.attrs else 1, 1)
				for _ in range(col_span):
					for ri in range(row_span):
						grid_table[(row_idx + ri, col_idx)] = col.text
					col_idx += 1
			row_idx += 1
		return cell_start_indices

	table_cell_start_indices = get_table_cell_start_indices(table, start_row_idx=0)
	#if table.thead:
	#	table_cell_start_indices = get_table_cell_start_indices(table.thead, start_row_idx=0)
	#if table.tbody:
	#	table_cell_start_indices = get_table_cell_start_indices(table.tbody, start_row_idx=0)
	#if table.tfoot:
	#	table_cell_start_indices = get_table_cell_start_indices(table.tfoot, start_row_idx=0)

	print("Table cell start indices = {}.".format(table_cell_start_indices))

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
<tr><td>A</td><td>B</td><td colspan=2>C</td><td>D</td></tr>
</thead>
<tbody>
<tr><td>10</td><td>11</td><td>12</td><td rowspan=2>13</td><td>14</td></tr>
<tr><td rowspan=2 colspan=3>20</td><td>21</td></tr>
<tr><td rowspan=2>30</td><td>31</td></tr>
<tr><td>40</td><td>41</td><td>42</td><td>43</td></tr>
<tr><td>50</td><td>51</td><td>52</td><td>53</td><td>54</td></tr>
</tbody>
"""

	#html_page = BeautifulSoup("<html><body><table>" + table_tags + "</table></body></html>", features="lxml").prettify()
	html_page = BeautifulSoup("<table>" + table_tags + "</table>", features="lxml").prettify()
	#html_page = construct_html_table_page(table_tags)
	#print(html_page)

	# Show a HTML image.
	html_image = imgkit.from_string(html_page, False)
	html_image = PIL.Image.open(io.BytesIO(html_image))
	#plt.figure()
	plt.imshow(html_image)
	plt.tight_layout()
	#plt.title("HTML Table Image")
	plt.axis("off")
	plt.show()

def main():
	#table_test()

	index_table_cells()
	#visualize_table()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
