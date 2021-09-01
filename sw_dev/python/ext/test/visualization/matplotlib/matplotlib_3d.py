#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html

import numpy as np
#from mpl_toolkits import mplot3d
#import matplotlib.pylab as plt
import matplotlib.pyplot as plt

#%matplotlib inline
#%matplotlib notebook

def basic_example():
	fig = plt.figure()
	ax = plt.axes(projection='3d')

	# Data for a three-dimensional line.
	zline = np.linspace(0, 15, 1000)
	xline = np.sin(zline)
	yline = np.cos(zline)
	ax.plot3D(xline, yline, zline, 'gray')

	# Data for three-dimensional scattered points.
	zdata = 15 * np.random.random(100)
	xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
	ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
	ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');

def f(x, y):
	return np.sin(np.sqrt(x**2 + y**2))

def wireframe_example():
	x = np.linspace(-6, 6, 30)
	y = np.linspace(-6, 6, 30)

	X, Y = np.meshgrid(x, y)
	Z = f(X, Y)

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot_wireframe(X, Y, Z, color='black')
	ax.set_title('wireframe');

def surface_example():
	x = np.linspace(-6, 6, 30)
	y = np.linspace(-6, 6, 30)

	X, Y = np.meshgrid(x, y)
	Z = f(X, Y)

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
	ax.set_title('surface')

def contour_example():
	x = np.linspace(-6, 6, 30)
	y = np.linspace(-6, 6, 30)

	X, Y = np.meshgrid(x, y)
	Z = f(X, Y)

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.contour3D(X, Y, Z, 50, cmap='binary')
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	ax.view_init(60, 35)
	fig

def partial_polar_grid():
	r = np.linspace(0, 6, 20)
	theta = np.linspace(-0.9 * np.pi, 0.8 * np.pi, 40)
	r, theta = np.meshgrid(r, theta)

	X = r * np.sin(theta)
	Y = r * np.cos(theta)
	Z = f(X, Y)

	ax = plt.axes(projection='3d')
	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none');

def main():
	basic_example()

	wireframe_example()
	surface_example()
	contour_example()

	partial_polar_grid()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
