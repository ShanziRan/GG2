
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

def result_attenuation():
	real = source.photon('100kVp, 2mm Al')
	fake = fake_source(material.mev, 0.1, material.coeff('Aluminium'), 2)
	fake_ideal = fake_source(material.mev, 0.1, method='ideal')

	water = material.coeff('Water')
	bone = material.coeff('Bone')

	depths = np.arange(0, 10.1, 0.1)

	plot(np.log(ct_detect(real, water, depths)), label='Real, Water')
	plot(np.log(ct_detect(fake, water, depths)), label='Fake, Water')
	plot(np.log(ct_detect(fake_ideal, water, depths)), label='Fake(ideal), Water')
	plot(np.log(ct_detect(real, bone, depths)), label='Real, Bone')
	plot(np.log(ct_detect(fake, bone, depths)), label='Fake, Bone')
	plot(np.log(ct_detect(fake_ideal, bone, depths)), label='Fake(ideal), Bone')

	plt.legend()
	plt.xlabel('Depth (cm)')
	plt.ylabel('Log Attenuation')
	plt.show() # need to alter function in ct_lib

def result_sinogram():
    p1 = ct_phantom(material.name, 256, 1) # simple disk
    p2 = ct_phantom(material.name, 256, 2) # impulse at centre
    p3 = ct_phantom(material.name, 256, 9) # impulse at different location
    p4 = ct_phantom(material.name, 256, 4) # bilateral hip replacement

    ps = [p1, p2, p3, p4]
    s = fake_source(source.mev, 0.1, method='ideal')
    for i, p in enumerate(ps):
        sinogram = ct_scan(s, material, p, 0.1, 256)

        # Save the sinogram for each phantom
        save_draw(sinogram, 'reports/results', f'sinogram_{i+1}')

def result_sinogram_angle():
	p = ct_phantom(material.name, 256, 4)
	s = fake_source(source.mev, 0.1, method='ideal')
	angles = [128, 512]

	for angle in angles:
		sinogram = ct_scan(s, material, p, 0.1, angle)
		save_draw(sinogram, 'reports/results', f'sinogram_angle_{angle}')

def result_sinogram_calibration():
	p = ct_phantom(material.name, 256, 4)
	s = fake_source(source.mev, 0.1, method='ideal')

	sinogram = ct_scan(s, material, p, 0.1, 256)
	sinogram_calibrated = ct_calibrate(s, material, sinogram, 0.1)

	save_draw(sinogram_calibrated, 'reports/results', 'sinogram_calibrated')

def result_save():
	p = ct_phantom(material.name, 256, 1)
	s = fake_source(source.mev, 0.1, method='ideal')
	y = scan_and_reconstruct(
		s, 
		material, 
		p, 
		scale=0.1, 
		angles=256, 
		mas=1000,
		alpha=0.0001, 
		background_mean=0.0,
		scatter_fraction=0.3
	)

	save_draw(y, 'reports/results', 'reconstruct_filter_disk', caxis=[0., np.max(y)])

def result_reconstruction_angle():
	p = ct_phantom(material.name, 256, 2, "Titanium")
	s = fake_source(source.mev, 0.1, method='ideal')
	angles = [256]

	for angle in angles:
		y1 = scan_and_reconstruct(
			s, 
			material, 
			p, 
			scale=0.1, 
			angles=angle, 
			mas=10000,
			alpha=5, 
			background_mean=0.0,
			scatter_fraction=0.3
		)
		y2 = scan_and_reconstruct(
			s, 
			material, 
			p, 
			scale=0.1, 
			angles=angle, 
			mas=10000,
			alpha=0.0001, 
			background_mean=0.0,
			scatter_fraction=0.3
		)
		# draw(y, caxis=[0., np.max(y)])
		print(p[127].max(), y1.max(), y2.max())
		plt.plot(p[127] * 0.02, label='Delta Phantom')
		plt.plot(y1[128] * 0.9, label='Ram-Lak+Raised Cosine (alpha=5)')
		plt.plot(y2[128] * 0.2, label='Ram-Lak')
		plt.xlim(85, 170)
		plt.legend()
		plt.show()
# Run the various tests
result_reconstruction_angle()