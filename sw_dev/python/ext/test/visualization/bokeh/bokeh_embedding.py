#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >>
#	https://bokeh.pydata.org/en/latest/docs/user_guide/embed.html
#	http://biobits.org/bokeh-flask.html

from flask import Flask, render_template, request
from bokeh.plotting import figure
from bokeh.embed import components
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the Iris dataset.
iris_df = pd.read_csv('data/iris.csv', names=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Species'], header=0)
feature_names = iris_df.columns[0:-1].values.tolist()

# Create the main plot.
def create_figure(current_feature_name, bins):
	hist, bin_edges = np.histogram(iris_df[current_feature_name].values, bins=bins, density=True)

	p = figure(plot_width=600, plot_height=400, title=current_feature_name, toolbar_location=None, tools='')
	p.quad(top=hist, bottom=0, left=bin_edges[:-1], right=bin_edges[1:], fill_color="#036564", line_color="#033649")

	p.legend.location = 'top_right'
	# Set the x axis label.
	p.xaxis.axis_label = current_feature_name
	# Set the y axis label.
	p.yaxis.axis_label = 'Count'

	return p

# Iris index page.
@app.route('/iris', methods=['GET'], strict_slashes=False)
def iris_index():
	print('****************** Entered iris_index().')

	# Determine the selected feature.
	current_feature_name = request.args.get('feature_name')
	if current_feature_name is None:
		current_feature_name = 'Sepal Length'

	# Create the plot.
	plot = create_figure(current_feature_name, 10)

	# Embed plot into HTML via Flask Render.
	script, div = components(plot)
	return render_template('iris_index.html',
		iris_script=script, iris_div=div,
		iris_feature_names=feature_names, iris_current_feature_name=current_feature_name
	)

# Hello index page, no args.
@app.route('/', methods=['GET'], strict_slashes=False)
def hello_index():
	name = request.args.get('name')
	if name is None:
		name = 'Flask'

	return render_template('hello_index.html', name=name)

def main():
	# With debug=True, Flask server will auto-reload when there are code changes.
	app.run(port=5000, debug=True)

#%%------------------------------------------------------------------

# Usage:
#	python bokeh_embedding.py

if '__main__' == __name__:
	main()
