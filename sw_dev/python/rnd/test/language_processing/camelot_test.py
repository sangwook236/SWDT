#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import camelot
from IPython.display import display
import matplotlib.pyplot as plt

# REF [site] >> https://camelot-py.readthedocs.io/en/master/user/quickstart.html
def quick_start_example():
	pdf_filepath = "/path/to/sample.pdf"

	try:
		tables = camelot.read_pdf(pdf_filepath)
		#tables = camelot.read_pdf(pdf_filepath, pages="1,2,3")  # 1-based indexing.
		#tables = camelot.read_pdf(pdf_filepath, password="password")
	except IOError as ex:
		print("File not found, {}: {}.".format(pdf_filepath, ex))
		return

	print("type(tables) = {}, len(tables) = {}.".format(type(tables), len(tables)))  # camelot.core.TableList.
	print("type(table) = {}.".format(type(tables[0])))  # camelot.core.Table.

	if len(tables) > 0:
		table = tables[0]

		print("table.parsing_report: {}.".format(table.parsing_report))
		print("table.accuracy = {}.".format(table.accuracy))  # Accuracy with which text was assigned to the cell.
		print("table.whitespace = {}.".format(table.whitespace))  # Percentage of whitespace in the table.
		print("table.order = {}.".format(table.order))  # Table number on PDF page.
		print("table.page = {}.".format(table.page))  # PDF page number.
		print("table.flavor = {}.".format(table.flavor))  # {"lattice", "stream"}.

		print("table.shape = {}.".format(table.shape))  # (rows, columns).
		print("table.rows = {}.".format(table.rows))  # List of tuples representing row y-coordinates in decreasing order.
		print("table.cols = {}.".format(table.cols))  # List of tuples representing column x-coordinates in increasing order.

		print("table.data = {}.".format(table.data))  # Cell texts. Row-wise grouping.
		print("table.cells = {}.".format(table.cells))  # Cell bboxes. Row-wise grouping. A list of lists of camelot.core.Cell's.

		cell = table.cells[0][0]
		print("Cell:")
		print("\t(x1, y1, x2, y2) = ({}, {}, {}, {}).".format(cell.x1, cell.y1, cell.x2, cell.y2))  # (left, bottom, right, top).
		print("\tlb = {}, lt = {}, rb = {}, rt = {}.".format(cell.lb, cell.lt, cell.rb, cell.rt))
		print("\t(left, right, top, bottom) = ({}, {}, {}, {}).".format(cell.left, cell.right, cell.top, cell.bottom))  # Whether or not cell is bounded on the left/right/top/bottom.
		print("\thspan = {}, vspan = {}.".format(cell.hspan, cell.vspan))  # Whether or not cell spans horizontally/vertically.
		print("\ttext = {}.".format(cell.text))  # Text assigned to cell.

		print("table._bbox = {}.".format(table._bbox))  # Table bbox. (?)
		print("table._image (len = {}):.".format(len(table._image))) # ?
		print("\tShape = {}, dtype = {}.".format(table._image[0].shape, table._image[0].dtype))
		print("\tKeys = {}.".format(table._image[1].keys()))
		print("table._segments (len = {}) = {}.".format(len(table._segments), table._segments))  # ?
		print("table._text (len = {}) = {}.".format(len(table._text), table._text))  # Text bboxes. (?)

		#tables.export("./table.csv", f="csv", compress=False)
		table.to_csv("./table.csv")
		#table.to_excel("./table.xlsx")
		#table.to_html("./table.html")
		#table.to_json("./table.json")
		#table.to_markdown("./table.md")
		#table.to_sqlite("./table.db")

		display(table.df)
	else:
		print("No table found.")

# REF [site] >> https://camelot-py.readthedocs.io/en/master/user/advanced.htm
def advanced_usage():
	pdf_filepath = "/path/to/sample.pdf"

	try:
		tables = camelot.read_pdf(pdf_filepath)
		#tables = camelot.read_pdf(pdf_filepath, process_background=True)

		# Specify table areas.
		# 	It is useful to specify exact table boundaries.
		#tables = camelot.read_pdf(pdf_filepath, flavor="stream", table_areas=["316,499,566,337"])  # (left, top, right, bottom). PDF coordinate system.
		# Specify table regions.
		#	Tables might not lie at the exact coordinates every time but in an approximate region.
		#tables = camelot.read_pdf(pdf_filepath, table_regions=["170,370,560,270"])  # (left, top, right, bottom). PDF coordinate system.
		# Specify column separators.
		#tables = camelot.read_pdf(pdf_filepath, flavor="stream", columns=["72,95,209,327,442,529,566,606,683"])
		#tables = camelot.read_pdf(pdf_filepath, flavor="stream", columns=["72,95,209,327,442,529,566,606,683"], split_text=True)

		# Flag superscripts and subscripts.
		#tables = camelot.read_pdf(pdf_filepath, flavor="stream", flag_size=True)
		# Strip characters from text.
		#tables = camelot.read_pdf(pdf_filepath, flavor="stream", strip_text=" .\n")

		# Improve guessed table areas.
		#tables = camelot.read_pdf(pdf_filepath, flavor="stream", edge_tol=500)
		# Improve guessed table rows.
		#tables = camelot.read_pdf(pdf_filepath, flavor="stream", row_tol=10)
		# Detect short lines.
		#tables = camelot.read_pdf(pdf_filepath, line_scale=40)

		# Shift text in spanning cells.
		#tables = camelot.read_pdf(pdf_filepath, line_scale=40, shift_text=[""])
		#tables = camelot.read_pdf(pdf_filepath, line_scale=40, shift_text=["r", "b"]))
		# Copy text in spanning cells.
		#tables = camelot.read_pdf(pdf_filepath, copy_text=["v"]))

		# Tweak layout generation
		#	Camelot is built on top of PDFMiner's functionality of grouping characters on a page into words and sentences.
		# 	To deal with such cases, you can tweak PDFMiner's LAParams kwargs to improve layout generation, by passing the keyword arguments as a dict using layout_kwargs in read_pdf().
		#tables = camelot.read_pdf(pdf_filepath, layout_kwargs={"detect_vertical": False}))

		# Use alternate image conversion backends.
		#	When using the Lattice flavor, Camelot uses ghostscript to convert PDF pages to images for line recognition.
		#tables = camelot.read_pdf(pdf_filepath, backend="ghostscript"))  # {"ghostscript", "poppler"}.
	except IOError as ex:
		print("File not found, {}: {}.".format(pdf_filepath, ex))
		return

	# REF [site] >> https://camelot-py.readthedocs.io/en/master/api.html
	#camelot.handlers.PDFHandler class.
	#camelot.parsers.Stream class.
	#camelot.parsers.Lattice class.

	# Visualize.
	if len(tables) > 0:
		table = tables[0]

		camelot.plot(table, kind="text").show()
		plt.title("Table Text")
		camelot.plot(table, kind="grid").show()
		plt.title("Table Grid")
		camelot.plot(table, kind="contour").show()
		plt.title("Table Contour")
		if table.flavor == "lattice":
			camelot.plot(table, kind="line").show()
			plt.title("Table Line")
			camelot.plot(table, kind="joint").show()
			plt.title("Table Joint")
		if table.flavor == "stream":
			camelot.plot(table, kind="textedge").show()
			plt.title("Table TextEdge")

		plt.show()
	else:
		print("No table found.")

def main():
	quick_start_example()
	#advanced_usage()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
