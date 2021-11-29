# Galsyn
This library is designed for generating synthetic galaxy images. It has methods for including noise from the sky and detector, foreground stars, and galaxies. Spiral arms can also be added as a perturbation to the disk flux, while multivariate copulas are used to model parameter dependencies. The structure is organized as follows:

```
.
└── galsyn
    ├── __file__
    │   ├── field_dictionary_sdss.json
    │   ├── field_generator_sdss.pkl
    │   ├── gadotti_2009_bar_bulge_disk.pkl
    │   ├── gadotti_2009_bulge_disk.pkl
    │   ├── gadotti_2009_disk.pkl
    │   ├── mendez-abreu_2017_bar_bulge_disk_dbreak.pkl
    │   ├── mendez-abreu_2017_bar_bulge_disk.pkl
    │   ├── mendez-abreu_2017_bar_disk_dbreak.pkl
    │   ├── mendez-abreu_2017_bar_disk.pkl
    │   ├── mendez-abreu_2017_bulge_disk_dbreak.pkl
    │   ├── mendez-abreu_2017_bulge_disk.pkl
    │   ├── mendez-abreu_2017_disk_dbreak.pkl
    │   ├── mendez-abreu_2017_disk.pkl
    │   └── star_generator_sdss.pkl
    ├── galaxy
    │   ├── convolution.py
    │   ├── copula.py
    │   ├── dataset.py
    │   ├── dust.py
    │   ├── geometry.py
    │   ├── photometric.py
    │   ├── procedural.py
    │   ├── profile.py
    │   ├── signal.py
    │   ├── spiral
    │   │   ├── interpolate.py
    │   │   ├── logarithmic.py
    │   │   ├── modulate.py
    │   │   ├── pattern.py
    │   │   ├── profile.py
    │   │   └── spiral.py
    │   └── utils.py
    ├── image_generator.py
    ├── random
    │   ├── perlin.py
    │   ├── sample.py
    │   └── simplex.py
    ├── sky_detector.py
    ├── star
    │   ├── diffraction.py
    │   ├── generator.py
    │   └── psf.py
    └── utils.py
```

## Sky-Detector
The sky-detector copula (field_generator_sdss.pkl) is constructed by randomly sampling 5000 entries from the SDSS [Field](http://skyserver.sdss.org/dr7/en/help/browser/browser.asp?n=Field&t=U) table. It generates parameters related to the sky background, detector noise, and the point-spread function for each of the *ugriz* filter bands.

## Star
The star copula (star_generator_sdss.pkl) was constructed by randomly sampling 5000 entries from the SDSS [Star](http://skyserver.sdss.org/dr7/en/help/browser/browser.asp?n=Star&t=U) table. It generates PSF magnitudes for each the *ugriz* filter bands.

## Galaxy
The galaxy copulas were constructed using the data from [Gadotti (2009)](https://ui.adsabs.harvard.edu/abs/2009MNRAS.393.1531G/abstract) and [Mendez-Abreu et al (2017)](https://ui.adsabs.harvard.edu/abs/2017A%26A...598A..32M/abstract). A linear interpolation was performed to extend parameters values to the *u* and *z* bands.