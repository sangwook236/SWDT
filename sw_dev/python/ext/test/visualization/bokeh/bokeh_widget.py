#!/usr/bin/env python

from bokeh.io import output_file, show
from bokeh.models import Button
from bokeh.models.widgets import Dropdown, MultiSelect
from bokeh.models.callbacks import CustomJS
from bokeh import events
from bokeh.layouts import widgetbox, column
from bokeh.plotting import figure, curdoc

# REF [site] >> https://bokeh.pydata.org/en/latest/docs/user_guide/interaction/widgets.html

def button_example(use_bokeh_server=False):
	if not use_bokeh_server:
		output_file('button.html')

	button = Button(label='Press Me')
	if use_bokeh_server:
		# To use real Python callbacks, a Bokeh server application may be used.
		def button_python_callback(window=None):
			print('Button is cicked.')

		button.on_click(button_python_callback)
	else:
		# Only JavaScript callbacks may be used with standalone output.
		button_javascript_callback = CustomJS(code="""
			console.log('Click event occured at x-position: ')
		""")

		button.js_on_event(events.ButtonClick, button_javascript_callback)

	layout = column(button)
	if use_bokeh_server:
		curdoc().add_root(layout)
	else:
		show(layout)

def dropdown_example():
	output_file('dropdown.html')

	menu = [('Item 1', 'item_1'), ('Item 2', 'item_2'), None, ('Item 3', 'item_3')]
	dropdown = Dropdown(label='Dropdown button', button_type='warning', menu=menu)

	show(widgetbox(dropdown))

def multi_select_example():
	output_file('multi_select.html')

	multi_select = MultiSelect(title='Option:', value=['foo', 'quux'],
		options=[('foo', 'Foo'), ('bar', 'BAR'), ('baz', 'bAz'), ('quux', 'quux')]
	)

	show(widgetbox(multi_select))

def main():
	button_example(False)
	#dropdown_example()
	#multi_select_example()

# REF [site] >> http://bokeh.pydata.org/en/latest/docs/user_guide/server.html
def main_for_bokeh_server():
	print('***** Using Bokeh server *****')
	button_example(True)

#%%------------------------------------------------------------------

# Usage:
#	python bokeh_widget.py
#	bokeh serve --show bokeh_widget.py

if '__main__' == __name__:
	main()
elif 'bk_script_' in __name__:
	main_for_bokeh_server()
