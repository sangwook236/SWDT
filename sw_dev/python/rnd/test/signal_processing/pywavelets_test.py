#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math
import numpy as np
import pywt
import pywt.data
import matplotlib.pyplot as plt

# REF [site] >> https://pywavelets.readthedocs.io/en/latest/regression/wavelet.html
def basic_example():
	print('Family =', pywt.families())
	for family in pywt.families():
		print('\t%s family: ' % family + ', '.join(pywt.wavelist(family)))

	#--------------------
	w = pywt.Wavelet('db3')

	print('Summary =', w)
	print('Name = {}, short family name = {}, family name = {}.'.format(w.name, w.short_family_name, w.family_name))

	# Decomposition (dec_len) and reconstruction (rec_len) filter lengths.
	print('dec_len = {}, rec_len = {}.'.format(int(w.dec_len), int(w.rec_len)))
	# Orthogonality and biorthogonality.
	print('Orthogonal = {}, biorthogonal = {}.'.format(w.orthogonal, w.biorthogonal))
	# Symmetry.
	print('Symmetry = {}.'.format(w.symmetry))
	# Number of vanishing moments for the scaling function phi (vanishing_moments_phi) and the wavelet function psi (vanishing_moments_psi) associated with the filters.
	print('vanishing_moments_phi = {}, vanishing_moments_psi = {}.'.format(w.vanishing_moments_phi, w.vanishing_moments_psi))

	# Lowpass and highpass decomposition filters and lowpass and highpass reconstruction filters.
	print('Filter bank? =', w.filter_bank == (w.dec_lo, w.dec_hi, w.rec_lo, w.rec_hi))

# REF [site] >> https://pywavelets.readthedocs.io/en/latest/regression/wavelet.html
def custom_wavelet_object_example():
	# Passing the filter bank object that implements the filter_bank attribute.
	# The attribute must return four filters coefficients.
	class MyHaarFilterBank(object):
		@property
		def filter_bank(self):
			return ([math.sqrt(2)/2, math.sqrt(2)/2], [-math.sqrt(2)/2, math.sqrt(2)/2], [math.sqrt(2)/2, math.sqrt(2)/2], [math.sqrt(2)/2, -math.sqrt(2)/2])

	my_wavelet = pywt.Wavelet('My Haar Wavelet', filter_bank=MyHaarFilterBank())

	print('Summary =', my_wavelet)

	# Passing the filters coefficients directly as the filter_bank parameter.
	my_filter_bank = ([math.sqrt(2)/2, math.sqrt(2)/2], [-math.sqrt(2)/2, math.sqrt(2)/2], [math.sqrt(2)/2, math.sqrt(2)/2], [math.sqrt(2)/2, -math.sqrt(2)/2])
	my_wavelet = pywt.Wavelet('My Haar Wavelet', filter_bank=my_filter_bank)

	print('Summary =', my_wavelet)

	my_wavelet.orthogonal = True
	my_wavelet.biorthogonal = True

	print('Summary =', my_wavelet)

# REF [site] >> https://pywavelets.readthedocs.io/en/latest/regression/wavelet.html
def wavefun_example():
	# The wavefun() method is used to approximate scaling function (phi) and wavelet function (psi) at the given level of refinement, based on the filters coefficients.

	# The number of returned values varies depending on the wavelet's orthogonality property.
	# For orthogonal wavelets the result is tuple with scaling function, wavelet function and xgrid coordinates.

	w = pywt.Wavelet('sym3')
	print('Orthogonal =', w.orthogonal)
	(phi, psi, x) = w.wavefun(level=5)

	# For biorthogonal (non-orthogonal) wavelets different scaling and wavelet functions are used for decomposition and reconstruction, and thus five elements are returned:
	#	decomposition scaling and wavelet functions approximations, reconstruction scaling and wavelet functions approximations, and the xgrid.

	w = pywt.Wavelet('bior1.3')
	print('Orthogonal =', w.orthogonal)
	(phi_d, psi_d, phi_r, psi_r, x) = w.wavefun(level=5)

# REF [site] >> https://pywavelets.readthedocs.io/en/latest/regression/modes.html
def signal_extention_mode_example():
	print('Mode =', pywt.Modes.modes)

	try:
		pywt.dwt([1,2,3,4], 'db2', 'invalid')
	except ValueError as ex:
		print('Invalid mode.')

	x = [1, 2, 1, 5, -1, 8, 4, 6]
	for mode_name in ['zero', 'constant', 'symmetric', 'reflect', 'periodic', 'smooth', 'periodization']:
		mode = getattr(pywt.Modes, mode_name)
		cA, cD = pywt.dwt(x, 'db2', mode)
		print('Mode: {} ({}).'.format(mode, mode_name))

	# The default mode is symmetric.
	cA, cD = pywt.dwt(x, 'db2')  # Approximation and detail coefficients.
	print('Single level reconstruction of signal =', pywt.idwt(cA, cD, 'db2'))

	cA, cD = pywt.dwt(x, 'db2', mode='symmetric')  # Approximation and detail coefficients.
	print('Single level reconstruction of signal =', pywt.idwt(cA, cD, 'db2'))

# REF [site] >> https://pywavelets.readthedocs.io/en/latest/regression/dwt-idwt.html
def discrete_wavelet_transform_example():
	x = [3, 7, 1, 1, -2, 5, 4, 6]
	cA, cD = pywt.dwt(x, 'db2')  # Approximation and detail coefficients.

	print('Approximation coefficients =', cA)
	print('Detail coefficients =', cD)

	print('Single level reconstruction of signal =', pywt.idwt(cA, cD, 'db2'))

	#--------------------
	# Pass a Wavelet object instead of the wavelet name and specify signal extension mode (the default is symmetric) for the border effect handling

	w = pywt.Wavelet('sym3')

	# the output coefficients arrays length depends not only on the input data length but also on the :class:Wavelet type (particularly on its filters length that are used in the transformation).
	# If you expected that the output length would be a half of the input data length, well, that's the trade-off that allows for the perfect reconstruction…
	cA, cD = pywt.dwt(x, wavelet=w, mode='constant')

	print('Approximation coefficients =', cA)
	print('Detail coefficients =', cD)

	# To find out what will be the output data size use the dwt_coeff_len() function.
	print('Length of coefficients =', int(pywt.dwt_coeff_len(data_len=len(x), filter_len=w.dec_len, mode='symmetric')))
	print('Length of coefficients =', int(pywt.dwt_coeff_len(len(x), w, 'symmetric')))
	print('Length of coefficients =', len(cA))

	# The periodization (periodization) mode is slightly different from the others.
	# It's aim when doing the DWT transform is to output coefficients arrays that are half of the length of the input data.

	# Knowing that, you should never mix the periodization mode with other modes when doing DWT and IDWT.
	# Otherwise, it will produce invalid results.

	cA, cD = pywt.dwt(x, wavelet=w, mode='periodization')
	print('Single level reconstruction of signal =', pywt.idwt(cA, cD, 'sym3', 'symmetric'))  # Invalid mode.
	print('Single level reconstruction of signal =', pywt.idwt(cA, cD, 'sym3', 'periodization'))

	# Passing None as one of the coefficient arrays parameters is similar to passing a zero-filled array.
	print('Single level reconstruction of signal =', pywt.idwt([1,2,0,1], None, 'db2', 'symmetric'))
	print('Single level reconstruction of signal =', pywt.idwt([1, 2, 0, 1], [0, 0, 0, 0], 'db2', 'symmetric'))
	print('Single level reconstruction of signal =', pywt.idwt(None, [1, 2, 0, 1], 'db2', 'symmetric'))
	print('Single level reconstruction of signal =', pywt.idwt([0, 0, 0, 0], [1, 2, 0, 1], 'db2', 'symmetric'))

	# Only one argument at a time can be None.
	try:
		print('Single level reconstruction of signal =', pywt.idwt(None, None, 'db2', 'symmetric'))
	except ValueError as ex:
		print('Invalid coefficient parameter.')

	# When doing the IDWT transform, usually the coefficient arrays must have the same size.
	try:
		pywt.idwt([1, 2, 3, 4, 5], [1, 2, 3, 4], 'db2', 'symmetric')
	except ValueError as ex:
		print('Invalid size of coefficients arrays.')

	# Not every coefficient array can be used in IDWT.
	try:
		pywt.idwt([1, 2, 4], [4, 1, 3], 'db4', 'symmetric')
	except ValueError as ex:
		print('Invalid coefficient arrays length.')

	print('Coefficient length =', int(pywt.dwt_coeff_len(1, pywt.Wavelet('db4').dec_len, 'symmetric')))

# REF [site] >> https://pywavelets.readthedocs.io/en/latest/regression/multilevel.html
def multilevel_dwt_decomposition_example():
	x = [3, 7, 1, 1, -2, 5, 4, 6]

	# Multilevel DWT decomposition.
	db1 = pywt.Wavelet('db1')
	cA3, cD3, cD2, cD1 = pywt.wavedec(x, db1)

	print('Approximation coefficients (level 3) =', cA3)
	print('Detail coefficients (level 3) =', cD3)
	print('Detail coefficients (level 2) =', cD2)
	print('Detail coefficients (level 1) =', cD1)

	# Multilevel IDWT reconstruction.
	coeffs = pywt.wavedec(x, db1)

	print('Signal reconstruction =', pywt.waverec(coeffs, db1))

	# Multilevel stationary wavelet transform (SWT) decomposition.
	x = [3, 7, 1, 3, -2, 6, 4, 6]
	(cA2, cD2), (cA1, cD1) = pywt.swt(x, db1, level=2)

	print('Approximation coefficients (level 1) =', cA1)
	print('Detail coefficients (level 1) =', cD1)
	print('Approximation coefficients (level 2) =', cA2)
	print('Detail coefficients (level 2) =', cD2)

	[(cA2, cD2)] = pywt.swt(cA1, db1, level=1, start_level=1)

	print('Approximation coefficients (level 2) =', cA2)
	print('Detail coefficients (level 2) =', cD2)

	coeffs = pywt.swt(x, db1)
	print('Coefficients =', coeffs)
	print('Length of coefficients =', len(coeffs))
	print('Max level =', pywt.swt_max_level(len(x)))

# REF [site] >>
#	https://pywavelets.readthedocs.io/en/latest/regression/wp.html
#	https://pywavelets.readthedocs.io/en/latest/regression/wp2d.html
def wavelet_packet_example():
	raise NotImplementedError

# REF [site] >> https://pywavelets.readthedocs.io/en/latest/
def simple_example():
	# Load image.
	original = pywt.data.camera()

	# Wavelet transform of image.
	coeffs2 = pywt.dwt2(original, 'bior1.3')

	# Plot approximation and details.
	titles = ['Approximation', 'Horizontal detail', 'Vertical detail', 'Diagonal detail']
	cA, (cH, cV, cD) = coeffs2
	fig = plt.figure(figsize=(12, 3))
	for i, a in enumerate([cA, cH, cV, cD]):
		ax = fig.add_subplot(1, 4, i + 1)
		ax.imshow(a, interpolation='nearest', cmap=plt.cm.gray)
		ax.set_title(titles[i], fontsize=10)
		ax.set_xticks([])
		ax.set_yticks([])

	fig.tight_layout()
	plt.show()

def main():
	#basic_example()
	#custom_wavelet_object_example()
	#wavefun_example()
	#signal_extention_mode_example()
	#discrete_wavelet_transform_example()
	#multilevel_dwt_decomposition_example()
	#wavelet_packet_example()  # Not yet implemented.

	simple_example()

#--------------------------------------------------------------------

if '__main__' == __name__:
	main()
