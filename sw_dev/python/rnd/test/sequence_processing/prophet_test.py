#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from prophet import Prophet

# REF [site] >> https://towardsdatascience.com/anomaly-detection-time-series-4c661f6f165f
def detect_anomaly():
	data_filepath = './sunspots.txt'
	data = np.loadtxt(data_filepath, float)

	data_as_frame = pd.DataFrame(data, columns=['Months', 'SunSpots'])
	print(data_as_frame.tail(10))

	data_as_frame['ds'] = data_as_frame['Months'].astype(int)
	print(data_as_frame.head())

	data_as_frame['time_stamp'] = data_as_frame.apply(lambda x: (pd.Timestamp('1749-01-01') + pd.DateOffset(months=int(x['ds']))), axis=1)
	clean_df = data_as_frame.drop(['Months', 'ds'], axis=1)
	print(clean_df.head())

	clean_df.columns = ['y', 'ds']

	#--------------------
	def fit_predict_model(dataframe, interval_width=0.99, changepoint_range=0.8):
		m = Prophet(
			daily_seasonality=False, yearly_seasonality=False, weekly_seasonality=False,
			seasonality_mode='multiplicative', 
			interval_width=interval_width,
			changepoint_range=changepoint_range
		)
		m = m.fit(dataframe)

		forecast = m.predict(dataframe)
		forecast['fact'] = dataframe['y'].reset_index(drop=True)

		print('Displaying Prophet plot')
		fig1 = m.plot(forecast)

		return forecast

	pred = fit_predict_model(clean_df)

	#--------------------
	def detect_anomalies(forecast):
		forecasted = forecast[['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()
		#forecast['fact'] = df['y']

		forecasted['anomaly'] = 0
		forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1
		forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = -1

		# Anomaly importances.
		forecasted['importance'] = 0
		forecasted.loc[forecasted['anomaly'] == 1, 'importance'] = \
			(forecasted['fact'] - forecasted['yhat_upper']) / forecast['fact']
		forecasted.loc[forecasted['anomaly'] == -1, 'importance'] = \
			(forecasted['yhat_lower'] - forecasted['fact']) / forecast['fact']

		return forecasted

	pred = detect_anomalies(pred)
	print(pred.head())

	#--------------------
	if True:
		import matplotlib.pyplot as plt

		normal = pred[pred.anomaly == 0]
		anomaly = pred[pred.anomaly != 0]

		plt.figure()
		plt.plot(normal.ds, normal.fact, 'bo', anomaly.ds, anomaly.fact, 'ro', markersize=2)
		plt.tight_layout()

		plt.show()

	if False:
		import json
		import altair as alt
		from altair.vega import v3
		from IPython.display import HTML

		vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3.SCHEMA_VERSION
		vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'
		vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION
		vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'
		noext = "?noext"

		paths = {
			'vega': vega_url + noext,
			'vega-lib': vega_lib_url + noext,
			'vega-lite': vega_lite_url + noext,
			'vega-embed': vega_embed_url + noext
		}

		workaround = """
		requirejs.config({{
			baseUrl: 'https://cdn.jsdelivr.net/npm/',
			paths: {}
		}});
		"""

		def add_autoincrement(render_func):
			# Keep track of unique <div/> IDs.
			cache = {}
			def wrapped(chart, id="vega-chart", autoincrement=True):
				if autoincrement:
					if id in cache:
						counter = 1 + cache[id]
						cache[id] = counter
					else:
						cache[id] = 0
					actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])
				else:
					if id not in cache:
						cache[id] = 0
					actual_id = id
				return render_func(chart, id=actual_id)
			# Cache will stay outside.
			return wrapped
					
		@add_autoincrement
		def render(chart, id="vega-chart"):
			chart_str = """
			<div id="{id}"></div><script>
			require(["vega-embed"], function(vg_embed) {{
				const spec = {chart};     
				vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);
				console.log("anything?");
			}});
			console.log("really...anything?");
			</script>
			"""
			return HTML(
				chart_str.format(
					id=id,
					chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)
				)
			)

		HTML("".join((
			"<script>",
			workaround.format(json.dumps(paths)),
			"</script>",
			"This code block sets up embedded rendering in HTML output and<br/>",
			"provides the function `render(chart, id='vega-chart')` for use below."
		)))

		def plot_anomalies(forecasted):
			interval = alt.Chart(forecasted).mark_area(interpolate='basis', color='#7FC97F').encode(
				x=alt.X('ds:T', title='date'),
				y='yhat_upper',
				y2='yhat_lower',
				tooltip=['yearmonthdate(ds)', 'fact', 'yhat_lower', 'yhat_upper']
			).interactive().properties(title='Anomaly Detection')

			fact = alt.Chart(forecasted[forecasted.anomaly == 0]).mark_circle(size=15, opacity=0.7, color='Black').encode(
				x='ds:T',
				y=alt.Y('fact', title='Sunspots'),
				tooltip=['yearmonthdate(ds)', 'fact', 'yhat_lower', 'yhat_upper']
			).interactive()

			anomalies = alt.Chart(forecasted[forecasted.anomaly != 0]).mark_circle(size=30, color='Red').encode(
				x='ds:T',
				y=alt.Y('fact', title='Sunspots'),
				tooltip=['yearmonthdate(ds)', 'fact', 'yhat_lower', 'yhat_upper'],
				size = alt.Size('importance', legend=None)
			).interactive()

			return render(alt.layer(interval, fact, anomalies).properties(width=870, height=450).configure_title(fontSize=20))

		plot_anomalies(pred)

def main():
	detect_anomaly()

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
