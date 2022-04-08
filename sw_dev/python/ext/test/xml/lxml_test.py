#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import lxml.html, lxml.etree

def table_test():
	table_tags = """
<table>
<tr><th>Event</th><th>Start Date</th><th>End Date</th></tr>
<tr><td>a</td><td>b</td><td>c</td></tr>
<tr><td>d</td><td>e</td><td>f</td></tr>
<tr><td>g</td><td>h</td><td>i</td></tr>
</table>
"""

	table = lxml.etree.HTML(table_tags).find("body/table")
	rows = iter(table)
	headers = [col.text for col in next(rows)]
	for row in rows:
		values = [col.text for col in row]
		print(dict(zip(headers, values)))

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

	table = lxml.etree.HTML(table_tags2).find("body/table")
	for child in table:
		print(child.tag)
		for row in child:
			print("{}: {}".format(row.tag, list(col.tag for col in row)))

def main():
	table_test()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
