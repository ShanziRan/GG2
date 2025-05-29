from ct_scan import *
from ct_calibrate import *
from ct_lib import *
from ramp_filter import *
from back_project import *
from hu import *

def scan_and_reconstruct(photons, material, phantom, scale, angles, mas=100, alpha=3, background_mean=50000, scatter_fraction=0.3):

	""" Simulation of the CT scanning process
		reconstruction = scan_and_reconstruct(photons, material, phantom, scale, angles, mas, alpha)
		takes the phantom data in phantom (samples x samples), scans it using the
		source photons and material information given, as well as the scale (in cm),
		number of angles, time-current product in mas, and raised-cosine power
		alpha for filtering. The output reconstruction is the same size as phantom."""


	# convert source (photons per (mas, cm^2)) to photons (function covered in ct_detect)
	photons = photons * mas * scale ** 2
	p_noisy = photons.copy()
	# p_noisy = np.zeros_like(photons)

	# statistical noise (Poisson) for source photons
	# for large mean values, the Poisson distribution can be approximated by a Gaussian
	# normal_mask = photons > 1000
	# p_noisy[normal_mask] = np.random.normal(photons[normal_mask], np.sqrt(photons[normal_mask]))

	# # for small mean values, the Poisson distribution is used directly
	# poisson_mask = (photons > 0) & (photons <= 1000)
	# p_noisy[poisson_mask] = np.random.poisson(photons[poisson_mask])

	# create sinogram from phantom data, with received detector values
	sinogram = ct_scan(p_noisy, material, phantom, scale, angles, background_mean, scatter_fraction)

	# convert detector values into calibrated attenuation values
	sinogram = ct_calibrate(p_noisy, material, sinogram, scale)

	# Ram-Lak
	sinogram = ramp_filter(sinogram, scale, alpha)

	# Back-projection
	reconstruction = back_project(sinogram)

	# convert to Hounsfield Units
	# reconstruction = hu(reconstruction, material)

	return reconstruction