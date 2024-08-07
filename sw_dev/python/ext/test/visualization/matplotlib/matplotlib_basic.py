#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://matplotlib.org/users/pyplot_tutorial.html

import numpy as np
#import matplotlib.pylab as plt
import matplotlib.pyplot as plt

#%matplotlib inline
#%matplotlib notebook

def basic():
	#---------------------------------------------------------------------
	# Dynamic rc settings.

	plt.rcParams

	#---------------------------------------------------------------------

	fig = plt.figure()

	plt.plot([1, 2, 3, 4])
	plt.ylabel('some numbers')
	plt.show()
	
	plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
	plt.axis([0, 6, 0, 20])
	plt.show()

	t = np.arange(0.0, 5.0, 0.2)
	# Red dashes, blue squares and green triangles.
	plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
	plt.show()

	fig.savefig('./test.png')
	plt.close(fig)

def control_line_properties():
	x = np.arange(5)
	y = x**2
	plt.plot(x, y, linewidth=2.0)
	plt.show()

	line, = plt.plot(x, y, '-')
	line.set_antialiased(False)  # Turn off antialising.
	plt.show()

	x1 = np.arange(10)
	y1 = x1**2
	x2 = np.arange(10)
	y2 = x2**3
	lines = plt.plot(x1, y1, x2, y2)
	# Use keyword args.
	plt.setp(lines, color='r', linewidth=2.0)
	# MATLAB style string value pairs.
	plt.setp(lines, 'color', 'b', 'linewidth', 2.0)
	plt.show()

def f(t):
	return np.exp(-t) * np.cos(2 * np.pi * t)

def work_with_multiple_figures_and_axes():
	t1 = np.arange(0.0, 5.0, 0.1)
	t2 = np.arange(0.0, 5.0, 0.02)

	plt.figure(1)
	plt.subplot(211)
	#ax = plt.subplot(211)
	#ax.axes.get_xaxis().set_visible(False)  # Hides ticks and tick labels.
	#ax.axes.get_yaxis().set_visible(False)  # Hides ticks and tick labels.
	plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
	
	plt.subplot(212)
	plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')
	plt.show()

	plt.figure(1)  # The first figure.
	plt.subplot(211)  # The first subplot in the first figure.
	plt.plot([1, 2, 3])
	plt.subplot(212)  # The second subplot in the first figure.
	plt.plot([4, 5, 6])

	plt.figure(2)  # A second figure.
	plt.plot([4, 5, 6])  # Create a subplot(111) by default.

	plt.figure(1)  # Figure 1 current; subplot(212) still current.
	plt.subplot(211)  # Make subplot(211) in figure1 current.
	plt.title('Easy as 1, 2, 3')  # Subplot 211 title.

def work_with_text():
	# Fix random state for reproducibility.
	np.random.seed(19680801)

	mu, sigma = 100, 15
	x = mu + sigma * np.random.randn(10000)

	# The histogram of the data.
	n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)

	plt.xlabel('Smarts')
	#t = plt.xlabel('my data', fontsize=14, color='red')
	plt.ylabel('Probability')
	plt.title('Histogram of IQ')
	#plt.title(r'$\sigma_i=15$')  # TeX.
	plt.text(60, .025, r'$\mu=100,\ \sigma=15$')  # TeX.
	plt.axis([40, 160, 0, 0.03])
	plt.grid(True)
	plt.show()

	# Annotate text.
	ax = plt.subplot(111)

	t = np.arange(0.0, 5.0, 0.01)
	s = np.cos(2 * np.pi * t)
	line, = plt.plot(t, s, lw=2)

	plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
		arrowprops=dict(facecolor='black', shrink=0.05),
	)

	plt.ylim(-2, 2)
	plt.show()

from matplotlib.ticker import NullFormatter  # Useful for 'logit' scale.

def logarithmic_and_other_nonlinear_axes():
	# Fix random state for reproducibility.
	np.random.seed(19680801)

	# Make up some data in the interval ]0, 1[.
	y = np.random.normal(loc=0.5, scale=0.4, size=1000)
	y = y[(y > 0) & (y < 1)]
	y.sort()
	x = np.arange(len(y))

	# Plot with various axes scales.
	plt.figure(1)

	# Linear.
	plt.subplot(221)
	plt.plot(x, y)
	plt.yscale('linear')
	plt.title('linear')
	plt.grid(True)

	# Log.
	plt.subplot(222)
	plt.plot(x, y)
	plt.yscale('log')
	plt.title('log')
	plt.grid(True)

	# Symmetric log.
	plt.subplot(223)
	plt.plot(x, y - y.mean())
	plt.yscale('symlog', linthreshy=0.01)
	plt.title('symlog')
	plt.grid(True)

	# Logit.
	plt.subplot(224)
	plt.plot(x, y)
	plt.yscale('logit')
	plt.title('logit')
	plt.grid(True)
	# Format the minor tick labels of the y-axis into empty strings with 'NullFormatter', to avoid cumbering the axis with too many labels.
	plt.gca().yaxis.set_minor_formatter(NullFormatter())
	# Adjust the subplot layout, because the logit one may take more space
	# than usual, due to y-tick labels like "1 - 10^{-3}"
	plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

	plt.show()

def bar_chart_example():
	langs = ('Python', 'C++', 'Java', 'Perl', 'Scala', 'Lisp')
	performance = [10, 8, 6, 4, 2, 1]
	x_pos = np.arange(len(langs))

	plt.bar(x_pos, performance, align='center', alpha=0.5)
	plt.xticks(x_pos, langs)
	plt.ylabel('Usage')
	plt.title('Programming language usage')

	plt.show()

# REF [site] >> https://matplotlib.org/3.1.1/gallery/statistics/hist.html
def histogram_example():
	# Generate data and plot a simple histogram
	N_points = 100000
	n_bins = 20

	# Generate a normal distribution, center at x=0 and y=5.
	x = np.random.randn(N_points)
	y = .4 * x + np.random.randn(100000) + 5

	fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

	# We can set the number of bins with the 'bins' kwarg.
	axs[0].hist(x, bins=n_bins)
	axs[1].hist(y, bins=n_bins)

	#--------------------
	# Update histogram colors.
	fig, axs = plt.subplots(1, 2, tight_layout=True)

	# N is the count in each bin, bins is the lower-limit of the bin.
	N, bins, patches = axs[0].hist(x, bins=n_bins)

	# We'll color code by height, but you could use any scalar.
	fracs = N / N.max()

	# we need to normalize the data to 0..1 for the full range of the colormap.
	norm = colors.Normalize(fracs.min(), fracs.max())

	# Now, we'll loop through our objects and set the color of each accordingly.
	for thisfrac, thispatch in zip(fracs, patches):
		color = plt.cm.viridis(norm(thisfrac))
		thispatch.set_facecolor(color)

	# We can also normalize our inputs by the total number of counts.
	axs[1].hist(x, bins=n_bins, density=True)

	# Now we format the y-axis to display percentage.
	axs[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))

def main():
	#basic()
	#control_line_properties()

	work_with_multiple_figures_and_axes()
	#work_with_text()

	#logarithmic_and_other_nonlinear_axes()

	bar_chart_example()
	histogram_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
