#!/usr/bin/env python

import numpy as np
import plotly
import plotly.graph_objs as go
import plotly.express as px  # A high-level data visualization API that produces fully-populated graph object figures in single function-calls.
import plotly.io as pio  # The low-level plotly.io.show function.
from plotly.subplots import make_subplots

# REF [site] >> https://plotly.com/python/creating-and-updating-figures/
def figures_example():
	# Figures as dictionaries.

	fig = dict({
		"data": [{
			"type": "bar",
			"x": [1, 2, 3],
			"y": [1, 3, 2]
		}],
		"layout": {"title": {"text": "A Figure Specified By Python Dictionary"}}
	})

	pio.show(fig)

	#--------------------
	# Figures as graph objects.

	fig = go.Figure(
		data=[go.Bar(x=[1, 2, 3], y=[1, 3, 2])],
		layout=go.Layout(
			title=go.layout.Title(text="A Figure Specified By A Graph Object")
		)
	)
	fig.show()

	dict_of_fig = dict({
		"data": [{
			"type": "bar",
			"x": [1, 2, 3],
			"y": [1, 3, 2]
		}],
		"layout": {"title": {"text": "A Figure Specified By A Graph Object With A Dictionary"}}
	})

	fig = go.Figure(dict_of_fig)
	fig.show()

	# Converting graph objects to dictionaries and JSON.
	fig = go.Figure(
		data=[go.Bar(x=[1, 2, 3], y=[1, 3, 2])],
		layout=go.Layout(height=600, width=800)
	)
	fig.layout.template = None  # Slim down the output.

	print("Dictionary Representation of A Graph Object:\n\n" + str(fig.to_dict()))
	print("\n\n")
	print("JSON Representation of A Graph Object:\n\n" + str(fig.to_json()))
	print("\n\n")

	#--------------------
	# Representing figures in Dash.

	from dash import Dash, dcc, html, Input, Output
	import plotly.express as px
	import json

	fig = px.line(
		x=["a", "b", "c"], y=[1, 3, 2],  # Replace with your own data source.
		title="sample figure", height=325
	)

	app = Dash(__name__)

	app.layout = html.Div([
		html.H4("Displaying figure structure as JSON"),
		dcc.Graph(id="graph", figure=fig),
		dcc.Clipboard(target_id="structure"),
		html.Pre(
			id="structure",
			style={
				"border": "thin lightgrey solid", 
				"overflowY": "scroll",
				"height": "275px"
			}
		),
	])

	@app.callback(
		Output("structure", "children"), 
		Input("graph", "figure")
	)
	def display_structure(fig_json):
		return json.dumps(fig_json, indent=2)

	app.run_server(debug=True)

# REF [site] >> https://plotly.com/python/creating-and-updating-figures/
def create_figures_example():
	df = px.data.iris()
	fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species", title="A Plotly Express Figure")

	# If you print the figure, you'll see that it's just a regular figure with data and layout.
	print(fig)

	fig.show()

	# Graph objects figure constructor.
	fig = go.Figure(
		data=[go.Bar(x=[1, 2, 3], y=[1, 3, 2])],
		layout=dict(title=dict(text="A Figure Specified By A Graph Object"))
	)
	fig.show()

	# Figure factories.
	#	Figure factories (included in plotly.py in the plotly.figure_factory module) are functions that produce graph object figures, often to satisfy the needs of specialized domains.
	import plotly.figure_factory as ff

	x1, y1 = np.meshgrid(np.arange(0, 2, .2), np.arange(0, 2, .2))
	u1 = np.cos(x1) * y1
	v1 = np.sin(x1) * y1

	fig = ff.create_quiver(x1, y1, u1, v1)
	fig.show()

	# Make subplots.
	fig = make_subplots(rows=1, cols=2)
	fig.add_trace(go.Scatter(y=[4, 2, 1], mode="lines"), row=1, col=1)
	fig.add_trace(go.Bar(y=[2, 1, 3]), row=1, col=2)
	fig.show()

# REF [site] >> https://plotly.com/python/creating-and-updating-figures/
def update_figures_example():
	# Adding traces.
	fig = go.Figure()
	fig.add_trace(go.Bar(x=[1, 2, 3], y=[1, 3, 2]))
	fig.show()

	df = px.data.iris()
	fig = px.scatter(
		df, x="sepal_width", y="sepal_length", color="species",
		title="Using The add_trace() method With A Plotly Express Figure"
	)
	fig.add_trace(
		go.Scatter(
			x=[2, 4],
			y=[4, 8],
			mode="lines",
			line=go.scatter.Line(color="gray"),
			showlegend=False
		)
	)
	fig.show()

	# Adding traces to subplots.
	fig = make_subplots(rows=1, cols=2)
	fig.add_trace(go.Scatter(y=[4, 2, 1], mode="lines"), row=1, col=1)
	fig.add_trace(go.Bar(y=[2, 1, 3]), row=1, col=2)
	fig.show()

	df = px.data.iris()
	fig = px.scatter(
		df, x="sepal_width", y="sepal_length", color="species", facet_col="species",
		title="Adding Traces To Subplots Witin A Plotly Express Figure"
	)
	reference_line = go.Scatter(
		x=[2, 4],
		y=[4, 8],
		mode="lines",
		line=go.scatter.Line(color="gray"),
		showlegend=False
	)
	fig.add_trace(reference_line, row=1, col=1)
	fig.add_trace(reference_line, row=1, col=2)
	fig.add_trace(reference_line, row=1, col=3)
	fig.show()

	# Add trace convenience methods.
	fig = make_subplots(rows=1, cols=2)
	fig.add_scatter(y=[4, 2, 1], mode="lines", row=1, col=1)
	fig.add_bar(y=[2, 1, 3], row=1, col=2)
	fig.show()

	# Magic underscore notation.
	fig = go.Figure(
		#data=[go.Scatter(y=[1, 3, 2], line=dict(color="crimson"))],
		#layout=dict(title=dict(text="A Graph Objects Figure Without Magic Underscore Notation"))
		data=[go.Scatter(y=[1, 3, 2], line_color="crimson")],
		layout_title_text="A Graph Objects Figure With Magic Underscore Notation"
	)
	fig.show()

	# Updating figure layouts.
	fig = go.Figure(data=go.Bar(x=[1, 2, 3], y=[1, 3, 2]))
	fig.update_layout(
		title_text="Using update_layout() With Graph Object Figures",
		title_font_size=30
	)
	fig.show()

	# Note that the following update_layout() operations are equivalent:
	#fig.update_layout(title_text="update_layout() Syntax Example", title_font_size=30)
	#fig.update_layout(title_text="update_layout() Syntax Example", title_font=dict(size=30))
	#fig.update_layout(title=dict(text="update_layout() Syntax Example"), font=dict(size=30))
	#fig.update_layout({"title": {"text": "update_layout() Syntax Example", "font": {"size": 30}}})
	#fig.update_layout(title=go.layout.Title(text="update_layout() Syntax Example", font=go.layout.title.Font(size=30)))

	# Updating traces.
	fig = make_subplots(rows=1, cols=2)
	fig.add_scatter(
		y=[4, 2, 3.5], mode="markers",
		marker=dict(size=20, color="LightSeaGreen"),
		name="a", row=1, col=1
	)
	fig.add_bar(
		y=[2, 1, 3],
		marker=dict(color="MediumPurple"),
		name="b", row=1, col=1
	)
	fig.add_scatter(
		y=[2, 3.5, 4], mode="markers",
		marker=dict(size=20, color="MediumPurple"),
		name="c", row=1, col=2
	)
	fig.add_bar(
		y=[1, 3, 2],
		marker=dict(color="LightSeaGreen"),
		name="d", row=1, col=2
	)
	#fig.update_traces(marker=dict(color="RoyalBlue"))
	#fig.update_traces(marker=dict(color="RoyalBlue"), selector=dict(type="bar"))
	#fig.update_traces(marker_color="RoyalBlue", selector=dict(marker_color="MediumPurple"))
	#fig.update_traces(marker=dict(color="RoyalBlue"), col=2)
	fig.show()

	df = px.data.iris()
	fig = px.scatter(
		df, x="sepal_width", y="sepal_length", color="species",
		facet_col="species", trendline="ols", title="Using update_traces() With Plotly Express Figures"
	)
	fig.update_traces(
		line=dict(dash="dot", width=4),
		selector=dict(type="scatter", mode="lines")
	)
	fig.show()

	# Overwrite existing properties when using update methods.
	fig = go.Figure(go.Bar(x=[1, 2, 3], y=[6, 4, 9], marker_color="red"))  # Will be overwritten below.
	fig.update_traces(overwrite=True, marker={"opacity": 0.4})
	fig.show()

	# Conditionally updating traces.
	df = px.data.iris()
	fig = px.scatter(
		df, x="sepal_width", y="sepal_length", color="species",
		title="Conditionally Updating Traces In A Plotly Express Figure With for_each_trace()"
	)
	fig.for_each_trace(
		lambda trace: trace.update(marker_symbol="square") if trace.name == "setosa" else (),
	)
	fig.show()

	# Updating figure axes.
	df = px.data.iris()
	fig = px.scatter(
		df, x="sepal_width", y="sepal_length", color="species",
		facet_col="species", title="Using update_xaxes() With A Plotly Express Figure"
	)
	fig.update_xaxes(showgrid=False)
	fig.show()

	# Chaining figure operations.
	df = px.data.iris()
	(px.scatter(
		df, x="sepal_width", y="sepal_length", color="species",
		facet_col="species", trendline="ols",
		title="Chaining Multiple Figure Operations With A Plotly Express Figure"
	)
	.update_layout(title_font_size=24)
	.update_xaxes(showgrid=False)
	.update_traces(
		line=dict(dash="dot", width=4),
		selector=dict(type="scatter", mode="lines"))
	).show()

	# Property assignment.
	fig = go.Figure(data=go.Bar(x=[1, 2, 3], y=[1, 3, 2]))
	fig.layout.title.text = "Using Property Assignment Syntax With A Graph Object Figure"
	fig.show()

	fig = go.Figure(data=go.Bar(x=[1, 2, 3], y=[1, 3, 2]))
	fig.data[0].marker.line.width = 4
	fig.data[0].marker.line.color = "black"
	fig.show()

	# What about Dash?
	fig = go.Figure()  # Or any Plotly Express function e.g. px.bar(...).
	#fig.add_trace(...)
	#fig.update_layout(...)

	import dash
	import dash_core_components as dcc
	import dash_html_components as html

	app = dash.Dash()
	app.layout = html.Div([
		dcc.Graph(figure=fig)
	])

	app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter

def main():
	#print("Version = {}.".format(plotly.__version__))

	figures_example()
	#create_figures_example()
	#update_figures_example()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
