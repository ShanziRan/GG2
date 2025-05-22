import math
import numpy as np
import numpy.matlib
from scipy.fftpack import fft, ifft, fftfreq

def ramp_filter(sinogram, scale, alpha=0.001):
	""" Ram-Lak filter with raised-cosine for CT reconstruction

	fs = ramp_filter(sinogram, scale) filters the input in sinogram (angles x samples)
	using a Ram-Lak filter.

	fs = ramp_filter(sinogram, scale, alpha) can be used to modify the Ram-Lak filter by a
	cosine raised to the power given by alpha."""

	# get input dimensions
	angles = sinogram.shape[0]
	n = sinogram.shape[1]

	#Set up filter to be at least twice as long as input
	m = np.ceil(np.log(2*n-1) / np.log(2))
	m = int(2 ** m)

	# create the raised-cosine combined ram-lak filter with the set length
	freqs = fftfreq(m, d=scale)
	ramlak = np.abs(freqs) * ((np.cos((np.pi * freqs) / (2 * freqs[-1]))) ** alpha)
	ramlak[0] = ramlak[1]/6


	# apply filter to all angles
	print('Ramp filtering')

	filtered = np.zeros((angles, n))
	for i in range(angles):
		proj_fft = fft(sinogram[i], m)
		proj_fft = proj_fft * ramlak
		proj_filtered = np.real(ifft(proj_fft))

        # Truncate back to original length
		filtered[i] = proj_filtered[:n]

	return filtered