"""
Methods for combining the galaxy, star, and sky-detector noise generators.

Classes
-------
ImageGenerator
    Class used for generating synthetic images, combining sky/detector
    noise, foreground stars, a galaxy, and background galaxies.

SyntheticDataset
    Pytorch dataset wrapper for the synthetic image generator.
"""

import numpy
import random
import torch
import torch.nn.functional as F
from typing import Iterable, Optional, Tuple, Union
from galkit.functional.transform import arcsinh_stretch
from .galaxy import BackgroundGalaxy, Gadotti, MendezAbreu, IsoFlux
from .sky_detector import SkyDetectorGenerator
from .star import StarGenerator, SDSSModel, DiffractionWithoutPhase
from .random import random_uniform
from .utils import load_local_generator

class ImageGenerator:
    """
    Class used for generating synthetic images, combining sky/detector
    noise, foreground stars, a galaxy, and background galaxies.

    Examples
    --------
    import matplotlib.pyplot as plt
    from galkit.functional import arcsinh_stretch
    from galsyn.image_generator import ImageGenerator

    im = ImageGenerator()
    output = im()

    fig, ax = plt.subplots()
    ax.imshow(arcsinh_stretch(output['image']).T)
    fig.show()
    """
    def __init__(self,
        arm_count : callable = lambda size, device : torch.multinomial(torch.tensor([2, 0, 1, 1, 1, 1], dtype=torch.float32, device=device), num_samples=size, replacement=True),
        device : Optional[torch.device] = None,
        filter_bands   : Union[callable, Iterable] = lambda : numpy.random.choice(['zir', 'irg', 'rgu']),
        isoA_scale     : Union[callable, float] = lambda size, device: random_uniform(1, 2, size, device),
        isoA_scale_bkg : Union[callable,float] = lambda size, device: random_uniform(3**2,10**2,size,device).sqrt(),
        isoA_metric : callable = IsoFlux(),
        isoA_value  : Optional[float] = None,
        GalaxyGenerators = (
            Gadotti(load_local_generator('gadotti_2009_bar_bulge_disk.pkl')),
            Gadotti(load_local_generator('gadotti_2009_bulge_disk.pkl')),
            Gadotti(load_local_generator('gadotti_2009_disk.pkl')),
            MendezAbreu(load_local_generator('mendez-abreu_2017_bar_bulge_disk.pkl')),
            MendezAbreu(load_local_generator('mendez-abreu_2017_bar_bulge_disk_dbreak.pkl')),
            MendezAbreu(load_local_generator('mendez-abreu_2017_bar_disk.pkl')),
            MendezAbreu(load_local_generator('mendez-abreu_2017_bulge_disk.pkl')),
            MendezAbreu(load_local_generator('mendez-abreu_2017_bulge_disk_dbreak.pkl')),
            MendezAbreu(load_local_generator('mendez-abreu_2017_disk.pkl')),
            MendezAbreu(load_local_generator('mendez-abreu_2017_disk_dbreak.pkl')),
        ),
        BkgGalaxyGenerator = BackgroundGalaxy(),
        galaxies_per_pixel : Union[callable,float] = 2e-5,
        plate_scale : Union[callable,float] = 0.396,
        output_arm_mask : bool = True,
        output_arm_s2n : bool = False,
        output_bar_mask : bool = False,
        output_bar_s2n : bool = False,
        output_galaxy_mask : bool = False,
        output_galaxy_s2n : bool = False,
        output_projection : bool = True,
        oversample : Union[callable, int] = 2,
        shape : Union[int, callable] = lambda : numpy.random.choice(numpy.arange(48,257,16)),
        SkyDetector = SkyDetectorGenerator(),
        StarGenerator = StarGenerator(psf_model=DiffractionWithoutPhase(SDSSModel(), width=0.1, fraction=0.90)),
        stars_per_pixel : Union[callable, float] = 4e-4,
    ):
        self.__dict__.update(**locals())
        self.SkyDetector.device = device
        self.StarGenerator.device = device
        for g in self.GalaxyGenerators:
            g.device = device
            g.spiral.arm_count = arm_count
        for g in self.BkgGalaxyGenerator.generators:
            g.device = device
            g.spiral.arm_count = arm_count

    def __call__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)

        galaxy_id = numpy.random.randint(len(self.GalaxyGenerators))
        Galaxy = self.GalaxyGenerators[galaxy_id]
        self.SkyDetector.sample(1)

        kwargs = {
            'filter_bands' : self.filter_bands() if callable(self.filter_bands) else self.filter_bands, 
            'oversample'   : self.oversample() if callable(self.oversample) else self.oversample,
            'plate_scale'  : self.plate_scale() if callable(self.plate_scale) else self.plate_scale,
            'shape'        : 2 * (self.shape() if callable(self.shape) else self.shape,), 
            'sky_detector' : self.SkyDetector,
        }

        # The potential exist that when equating the flux value with the background
        # flux that the brightness is never large enough resulting in an error. 
        # We use a try loop to catch such cases here and for the background galaxies.
        incomplete = True
        while incomplete:
            try:
                output_galaxy = Galaxy(
                    isoA_metric = self.isoA_metric,
                    isoA_scale  = self.isoA_scale,
                    isoA_value  = self.isoA_value,
                    output_arm_mask = self.output_arm_mask,
                    output_arm_s2n = self.output_arm_s2n,
                    output_bar_mask = self.output_bar_mask,
                    output_bar_s2n = self.output_bar_s2n,
                    output_galaxy_mask = self.output_galaxy_mask,
                    output_galaxy_s2n = self.output_galaxy_s2n,
                    output_projection = self.output_projection,
                    size = 1,
                    **kwargs
                )
                incomplete = False
            except Exception as e:
                print(e)

        incomplete = True
        while incomplete:
            try:
                output_bkg = self.BkgGalaxyGenerator(
                    size = 1,
                    galaxies_per_pixel = self.galaxies_per_pixel,
                    isoA_metric = self.isoA_metric,
                    isoA_scale  = self.isoA_scale_bkg,
                    isoA_value  = self.isoA_value,
                    **kwargs,
                )
                incomplete = False
            except Exception as e:
                print(e)

        output_star = self.StarGenerator(
            stars_per_pixel = self.stars_per_pixel() if callable(self.stars_per_pixel) else self.stars_per_pixel,
            **kwargs
        )

        output_skydetector = self.SkyDetector(
            shape = kwargs['shape'],
            filter_bands = kwargs['filter_bands'],
            plate_scale = kwargs['plate_scale'],
        )


        image = {}
        for k in kwargs['filter_bands']:
            image[k] = output_skydetector[k] \
                     + output_star['flux'][k] \
                     + output_galaxy['flux'][0][k] \
                     + output_bkg[0][k]
        
        return {
            'image': torch.cat([v for v in image.values()]),
            **{k:v for k,v in output_galaxy.items() if k != 'flux'},
            **{k:v for k,v in output_star.items() if k != 'flux'},
        }

class SyntheticDataset(ImageGenerator, torch.utils.data.Dataset):
    """
    Pytorch dataset wrapper for the synthetic image generator.

    Examples
    --------
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    from galkit.functional import to_tricolor, fits2jpeg
    from galsyn.image_generator import SyntheticDataset

    im = SyntheticDataset(size=100)
    output = im[0]

    image = output['image']
    mask = output['arm_mask']

    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(fits2jpeg(image).permute(1,2,0))
    if mask.size(0) == 0:
        ax[1].imshow(torch.zeros(256,256))
    else:
        ax[1].imshow(to_tricolor(mask, sns.color_palette('tab10')))
    fig.show()
    """
    def __init__(self,
        size : int,
        output_shape : Tuple[int,int] = (256,256),
        transform : Optional[callable] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        size : int
            The number of samples in a batch.

        output_shape : Tuple[int,int]
            The desired output shape. The output tensors are interpolated
            to the indicated shape.

        transform : callable, optional
            A function that takes as input the image and returns a
            transformed image.

        **kwargs
            Arguments to pass into the parent constructor.
        """
        super().__init__(**kwargs)
        self.size = size
        self.transform = transform
        self.ikeys = {
            'size': output_shape,
            'align_corners': True,
            'mode': 'bilinear',
        }

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx=None):
        sample = self()

        for k,v in sample.items():

            if k == 'projection':
                sample[k] = {kk:v[f'{kk}_disk'] for kk in ['h0', 'w0', 'pa', 'q']}
            elif k == 'image':
                sample['image'] = F.interpolate(v.unsqueeze(0), **self.ikeys).squeeze(0)
            else:
                if isinstance(v, (list, tuple)):
                    v = v[0]
                if isinstance(v, torch.Tensor):
                    sample[k] = v if v.size(0) == 0 else F.interpolate(v.unsqueeze(0), **self.ikeys).squeeze(0)
                else:
                    sample[k] = {kk:vv if vv.size(0) == 0 else F.interpolate(v.unsqueeze(0), **self.ikeys).squeeze(0) for kk,vv in v.items()}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample