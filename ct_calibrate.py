import numpy as np
from ct_detect import ct_detect
from scipy.interpolate import interp1d  # You can remove this if not needed elsewhere
from attenuate import attenuate

def ct_calibrate(photons, material, sinogram, scale):
    """ct_calibrate: Convert CT detections to linearised attenuation using polynomial fit."""
    
    # Step 1: Air-based normalisation
    n = sinogram.shape[1]
    air_idx = material.name.index('Air')
    air_coeff = material.coeffs[air_idx]
    depth = 2 * n * scale  # fixed source-detector distance
    source_total = attenuate(photons, air_coeff, depth)
    sinogram = -np.log(sinogram / np.sum(source_total))  # log-normalise

    # Step 2: Water-based beam hardening correction using polynomial fit
    depths = np.linspace(0, depth, 200)
    original_energy = np.tile(photons[:, np.newaxis], (1, 200))

    # Simulate attenuation through water
    water_idx = material.name.index('Water')
    water_coeff = material.coeffs[water_idx]
    residual_energy = attenuate(original_energy, water_coeff, depths)  # shape (len(photons), len(depths))

    # Compute total transmitted energy for each depth
    I_tot = np.sum(residual_energy, axis=0)  # shape (200,)
    I0 = np.sum(photons)  # scalar
    p_w = -np.log(I_tot / I0)  # shape (200,)

    # Fit a polynomial: depth = f(p_w)
    deg = 10  # You can change degree based on accuracy-vs-stability tradeoff
    coeffs = np.polyfit(p_w, depths, deg)

    # Evaluate the polynomial to get estimated depth for each sinogram value
    sinogram = np.polyval(coeffs, sinogram)

    return sinogram
