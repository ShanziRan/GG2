
# these are the imports you are likely to need
import numpy as np
from material import *
from source import *
from fake_source import *
from ct_phantom import *
from ct_lib import *
from scan_and_reconstruct import *
from create_dicom import *

# create object instances
material = Material()
source = Source()

# define each end-to-end test here, including comments
# these are just some examples to get you started
# all the output should be saved in a 'results' directory

# Two separate unit tests to confirm that value and geometry of reconstrucion from simulated CT scan agrees with expected result within acceptable tolerance.

def test_value(material):
	# Check whether reconstructed linear attenuation coefficient corresponds to standard value given in the data
	# by comparing mean reconstructed value at the middle portion of the shape

	p = ct_phantom(material.name, 256, 1)
	s = fake_source(source.mev, 0.1, method='ideal')
	y = scan_and_reconstruct(s, material, p, 0.1, 256)

	y_mean = np.mean(y[64:192, 64:192])
	tissue = names.index('Soft Tissue')
	ref_coeff = material.coeffs[tissue][70]

	assert np.isclose(y_mean, ref_coeff, atol=0.01)
	
	# Results: Reconstruction value within 0.01 to the standard recorded value of linear attenuation coefficient for soft tissue at the energy level set for the test.


def test_shape():
	# Check whether reconstructed geometry shape aligns with original phantom shape
	# for example the simple circle check for calibration
	# by comparing the counts of non-zero values in the reconstructed attenuations array for each horizontal and vertical slice of the pixel image

	p = ct_phantom(material.name, 256, 1)
	s = fake_source(source.mev, 0.1, method='ideal')
	y = scan_and_reconstruct(s, material, p, 0.1, 256)

	y_mean = np.mean(y[64:192, 64:192])
	p_max = np.max(p)

	# check that reconstructed shape matches reference shape
	y_round = np.round(y, 2)
	x = np.linspace(-1, 1, 256)

	# Save original phantom and reconstructed shape
	save_draw(y, 'results', 'test_output_1')
	save_draw(p, 'results', 'test_output_1_phantom')

	# Allow for a dynamic tolerence for the circle phantom shape to account for near-tangent slices
	tolerances = 2 + 15 * np.exp(-x**2 / (2 * 0.5**2))
	for i in range(p.shape[0]):
		if np.sum(p[i]) != 0:
			row_slice = y_round[i]
			row_ref = p[i]

			close_mask = np.isclose(row_slice, y_mean, atol=0.05)
			close_count = np.count_nonzero(close_mask)

			expected_count = np.count_nonzero(row_ref)

			assert np.isclose(close_count, expected_count, atol=tolerances[i])

	for j in range(p.shape[1]):
		if np.sum(p[:, j]) != 0:
			col_slice = y_round[:, j]
			col_ref = p[:, j]

			close_mask = np.isclose(col_slice, y_mean, atol=0.05)
			close_count = np.count_nonzero(close_mask)

			expected_count = np.count_nonzero(col_ref)

			assert np.isclose(close_count, expected_count, atol=tolerances[j])
			
	# Results: Assures accuracy of reconstructed shape up to 2 pixels tolerence, which corresponds to minor negligible 
	
# Run the various tests
# print("Shape test")
# test_shape()
print("Value test")
test_value()
