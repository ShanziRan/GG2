import numpy as np
from attenuate import *
from ct_calibrate import *
from ct_detect import *

def hu(p, material, reconstruction, scale):
	""" convert CT reconstruction output to Hounsfield Units
	calibrated = hu(p, material, reconstruction, scale) converts the reconstruction into Hounsfield
	Units, using the material coefficients, photon energy p and scale given."""
 
 #attenuate to get residual energy through water and then calibrate to get total attenuation coefficient
	print('p shape is', p.shape)
	n = reconstruction.shape[1]
	depth = 2 * n * scale  # for the fixed distance between source and detector
	water_idx = material.name.index('Water')
	water_coeff = material.coeffs[water_idx]
	water_energy = np.array(ct_detect(p, water_coeff, depth), ndmin=2)
	print('water_energy shape is', water_energy.shape)
	calibrated = ct_calibrate(p, material, water_energy, scale)
	mu_water = calibrated/depth
 
	print('mu_water shape is',mu_water.shape)
 
	hu = ((reconstruction-mu_water)/ mu_water) * 1000
	# g = ((hu-center)/width) * 128 +128
	# g = np.clip(g, 0, 255)
	clipped = np.clip(hu, -1024.0, 3072.0)
  
	# use water to calibrate

	# put this through the same calibration process as the normal CT data

	# use result to convert to hounsfield units
	# limit minimum to -1024, which is normal for CT data.

	return clipped