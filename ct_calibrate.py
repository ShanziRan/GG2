import numpy as np
import scipy
from scipy import interpolate
from attenuate import attenuate
from ct_detect import ct_detect

def ct_calibrate(photons, material, sinogram, scale):

	""" ct_calibrate convert CT detections to linearised attenuation
	sinogram = ct_calibrate(photons, material, sinogram, scale) takes the CT detection sinogram
	in x (angles x samples) and returns a linear attenuation sinogram
	(angles x samples). photons is the source energy distribution, material is the
	material structure containing names, linear attenuation coefficients and
	energies in mev, and scale is the size of each pixel in x, in cm."""

	# Get dimensions and work out detection for just air of twice the side
	# length (has to be the same as in ct_scan.py)
	n = sinogram.shape[1]

	# perform calibration
	# retrieve air coefficients
	air = material.name.index('Air')
	air_coeff = material.coeffs[air]

	depth = 2 * n * scale # for the fixed distance between source and detector
	print('depth',depth)
	# calculate air attenuated energy (calibration) for each energy level
	source_total = attenuate(photons, air_coeff, depth)

	# normalise the energy to total attenuation coefficient
	sinogram = - np.log(sinogram / sum(source_total))
	#beam hardening correction
	depths = np.arange(0, 100, 0.1)
	water_idx = material.name.index('Water')
	water_coeff = material.coeffs[water_idx]
	I_tot = ct_detect(photons, water_coeff, depths)
	p_w = -np.log(I_tot / np.sum(source_total))
	f = interpolate.interp1d(p_w, depths)
	sinogram = f(sinogram)
	return sinogram
