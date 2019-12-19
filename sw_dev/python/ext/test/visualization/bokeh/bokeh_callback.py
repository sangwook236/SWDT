#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, CustomJS, Slider
from bokeh import events
from bokeh.layouts import widgetbox, column
from bokeh.plotting import figure, curdoc
import math
from copy import deepcopy

# REF [site] >> http://bokeh.pydata.org/en/latest/docs/user_guide/interaction/callbacks.html

def simple_callback_example(use_bokeh_server=False):
	if not use_bokeh_server:
		output_file('simple_callback.html')

	x = [x * 0.005 for x in range(0, 200)]
	y = deepcopy(x)

	source = ColumnDataSource(data=dict(x=x, y=y))

	plot = figure(plot_width=400, plot_height=400)
	plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)
	figure_javascript_callback = CustomJS(code="""
		// The event that triggered the callback is cb_obj:
		// The event type determines the relevant attributes.
		console.log('Tap event occured at x-position: ' + cb_obj.x)
	""")

	# Execute a callback whenever the plot canvas is tapped.
	plot.js_on_event('tap', figure_javascript_callback)

	# The model that triggers the callback (i.e. the model that the callback is attached to) will be available as cb_obj.
	javascript_code = """
		var data = source.data;
		var f = cb_obj.value;
		var x = data['x'];
		var y = data['y'];
		for (var i = 0; i < x.length; i++)
		{
			y[i] = Math.pow(x[i], f);
		}
		source.change.emit();
		//source.trigger('change');  // Not working.
		console.log('Current value = ' + f);
	"""

	# The model that triggers the callback (i.e. the model that the callback is attached to) will be available as cb_obj.
	def slider_py2js_callback(source=source, window=None):
		data = source.data
		f = cb_obj.value
		x, y = data['x'], data['y']
		for i in range(len(x)):
			y[i] = window.Math.pow(x[i], f)
		source.change.emit()
		#source.trigger('change')  # Not working.

	def slider_python_callback(attr_name, old, new):
		data = source.data
		x, y = data['x'], data['y']
		for i in range(len(x)):
			y[i] = math.pow(x[i], new)
		#source.data = dict(x=x, y=y)
		data['y'] = y

	if False and use_bokeh_server:
		# To use real Python callbacks, a Bokeh server application may be used.
		slider = Slider(start=0.1, end=4, value=1, step=.1, title='power')
		slider.on_change('value', slider_python_callback)
	elif False:
		# Only JavaScript callbacks may be used with standalone output.
		slider = Slider(start=0.1, end=4, value=1, step=.1, title='power', callback=CustomJS.from_py_func(slider_py2js_callback))
	else:
		# Only JavaScript callbacks may be used with standalone output.
		slider_javascript_callback = CustomJS(args=dict(source=source), code=javascript_code)
		slider = Slider(start=0.1, end=4, value=1, step=.1, title='power', callback=slider_javascript_callback)
		#slider = Slider(start=0.1, end=4, value=1, step=.1, title='power')
		#slider.js_on_change('value', slider_javascript_callback)

	layout = column(slider, plot)
	if use_bokeh_server:
		curdoc().title = 'MotorSense Analytics'
		curdoc().add_root(layout)
	else:
		show(layout)

def main():
	simple_callback_example(False)

# REF [site] >> http://bokeh.pydata.org/en/latest/docs/user_guide/server.html
def main_for_bokeh_server():
	print('***** Using Bokeh server *****')
	simple_callback_example(True)

#%%------------------------------------------------------------------

# Usage:
#	python bokeh_callback.py
#	bokeh serve --show bokeh_callback.py

if '__main__' == __name__:
	main()
elif 'bk_script_' in __name__:
	main_for_bokeh_server()
