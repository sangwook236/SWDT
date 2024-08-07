#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from bokeh.io import output_file, show, save, export_png
from bokeh.layouts import column
from bokeh.plotting import figure
import pandas as pd
import numpy as np

# REF [site] >> https://bokeh.pydata.org/en/latest/docs/user_guide/plotting.html
def scatter_marker_example():
	# Output to static HTML file.
	output_file('scatter_marker.html')

	p = figure(plot_width=400, plot_height=400)

	# Add a circle renderer with a size, color, and alpha.
	p.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=20, color='red', fill_color='blue', alpha=0.5)
	p.square([1, 2, 3, 4, 5], [6, 7, 2, 3, 5], size=15, color='olive', alpha=0.5)

	# Show the results.
	show(p)

# REF [site] >> https://bokeh.pydata.org/en/latest/docs/user_guide/plotting.html
def line_example():
	# Output to static HTML file.
	output_file('line.html')

	p = figure(plot_width=400, plot_height=400)

	# Add a line renderer.
	p.line([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=2)
	# Add a steps renderer.
	p.step([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=2, line_dash='dashed', line_color='#1d91d0', mode='center')

	# Show the results.
	show(p)

def histogram_example():
	# Output to static HTML file.
	output_file('histogram.html')

	iris_df = pd.read_csv('data/iris.csv', names=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species'], header=0)
	#print(iris_df)
	
	#iris_df_setosa = iris_df.loc[iris_df['Species'] == 'Iris-setosa']
	#iris_df_versicolor = iris_df.loc[iris_df['Species'] == 'Iris-versicolor']
	#iris_df_virginica = iris_df.loc[iris_df['Species'] == 'Iris-virginica']

	current_feature_name = 'Petal Length'
	hist, bin_edges = np.histogram(iris_df[current_feature_name].values, bins=10, density=True)

	p = figure(plot_width=600, plot_height=400, title=current_feature_name, toolbar_location=None, tools='')
	p.quad(top=hist, bottom=0, left=bin_edges[:-1], right=bin_edges[1:], fill_color="#036564", line_color="#033649")
	#p.vbar(x=bin_edges, top=hist, width=0.9)

	p.legend.location = 'top_right'
	# Set the x axis label.
	p.xaxis.axis_label = current_feature_name
	# Set the y axis label.
	p.yaxis.axis_label = 'Count'

	# Show the results.
	show(p)

# REF [site] >> https://bokeh.pydata.org/en/latest/docs/user_guide/layout.html
def layout_example():
	output_file('layout.html')

	x = list(range(11))
	y0 = x
	y1 = [10 - i for i in x]
	y2 = [abs(i - 5) for i in x]

	# Create a new plot.
	s1 = figure(plot_width=250, plot_height=250, title=None)
	s1.circle(x, y0, size=10, color='navy', alpha=0.5)

	# Create another one.
	s2 = figure(plot_width=250, plot_height=250, title=None)
	s2.triangle(x, y1, size=10, color='firebrick', alpha=0.5)

	# Create and another.
	s3 = figure(plot_width=250, plot_height=250, title=None)
	s3.square(x, y2, size=10, color='olive', alpha=0.5)

	# Put the results in a column and show.
	show(column(s1, s2, s3))
	#save(column(s1, s2, s3))
	#export_png(column(s1, s2, s3), filename='./plot.png')

def main():
	#scatter_marker_example()
	#line_example()

	histogram_example()

	#layout_example()

#%%------------------------------------------------------------------

# Usage:
#	python bokeh_plot.py

if '__main__' == __name__:
	main()
