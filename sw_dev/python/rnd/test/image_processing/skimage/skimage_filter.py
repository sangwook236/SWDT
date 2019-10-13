#!/usr/bin/env python

import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import try_all_threshold

def try_all_threshold_example():
	img = data.page()

	fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
	plt.show()

def main():
	try_all_threshold_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
