"""
Perlin procedural noise.

Functions
---------
perlin_2d(shape, θ, r, resolution, corner_grid, device, fade,
         grid_base, grid_kwargs, repeat, rotation, transform)
    Generates a random perlin noise image.

"""

import math
import torch
from galkit.spatial import coordinate, grid
from typing import Optional, Tuple

def perlin2d(
    shape       : Optional[Tuple[int,int]] = None,
    θ           : Optional[torch.Tensor] = None,
    r           : Optional[torch.Tensor] = None,
    resolution  : Tuple[float,float] = (0.1, 0.1),
    corner_grad : bool = True,
    device      : Optional[torch.device] = None,
    fade        : callable = lambda t: 6*t**5 - 15*t**4 + 10*t**3,
    grid_base   : grid.Grid = grid.PytorchGrid(),
    grid_kwargs : dict = {},
    repeat      : int = 256,
    rotation    : Optional[callable] = None,
    shear       : float = 0.5,
    transform   : Optional[callable] = None,
):
    """
    Generates a random perlin noise image.

    Parameters
    ----------
    shape : Tuple[int,int], optional
        The shape of the image. Only needed if the parameters θ and
        r are set to None.
    
    θ : tensor, optional
        The azimuthal grid positions.

    r : tensor, optional
        The radial grid positions.

    resolution : tuple[float,float]
        The relative resolution compared to the image size. Features
        will have have a size comparable to the this fractional value
        of the image size.

    corner_grad : bool
        Boolean indicating whether to only consider gradients pointing
        from the origin of the four corner points (True) or whether any
        random gradient can be constructed (False). Default is True.

    device : torch.device, optional
        The device to generate data on. Only used if θ/r are not passed.
        Default is None.
    
    fade : callable
        Transition function between gradients.

    grid_base : Grid
        The Grid representing the base coordinate system. Used for
        converting grid values to pixels.

    grid_kwargs : dict
        Any kwargs used to construct the coordinate system.

    repeat : int
        The maximum number of grid points at which to repeat. Mostly
        used for memory purposes.

    rotation : callable, optional
        A function that takes as input the azimuth and radial positions
        are returns the azimuth rotation value.

    shear : callable, float
        The fractional value of the rotation to apply. A value of 1 will
        cause features to strongly follow the rotation pattern while a 
        value of zero will remove the rotational effect. Can be either a
        float or a function that returns a float. Only applied if a rotation
        parameter is supplied.

    transform : callable, optional
        Transformation operation to apply to the noise map output.

    Returns
    -------
    noise : tensor
        The perlin noise map. Values will be between -1 and +1 before
        any transformation function is applied.

    Examples
    --------
    import math
    import matplotlib.pyplot as plt
    from galkit.spatial import coordinate, grid
    from galsyn.random.perlin import perlin2d

    grid_base = grid.PytorchGrid()
    grid_kwargs = {
        'h0': 0.1,
        'w0': 0.1,
        'pa': math.pi/4,
        'q' : 0.5,
    }

    θ, r = coordinate.polar(
        grid = grid_base(128,128),
        **grid_kwargs
    )

    noise = perlin2d(θ=θ, r=r,
        grid_base = grid_base,
        grid_kwargs = grid_kwargs,
        resolution = (0.1, 0.1),
        rotation = lambda θ, r: r.add(1e-6).log().div(math.tan(math.pi/6)),
    )

    fig, ax = plt.subplots()
    ax.imshow(noise)
    ax.contour(r.squeeze(0))
    fig.show()

    To Do
    -----
    Generalize the rotation so that it can operate on both θ and r.
    """
    # Set-up the (x,y) coordinate system.
    if (θ is not None and r is not None) or grid_kwargs or rotation:
        if (θ is None) or (r is None):
            if shape is None:
                raise Exception("Must pass in either θ/r or shape parameter")

            θ, r = coordinate.polar(
                grid = grid_base(shape, device=device),
                **grid_kwargs,
            )

        # Apply a rotation
        if (shear > 0) and (rotation is not None):
            θ = θ - shear * rotation(θ=θ, r=r)

        # Undo any scaling to the coordinate system
        if 'scale' in grid_kwargs:
            r = r / grid_kwargs['scale']

        # Cast back to (x,y) coordinates
        x = r * θ.cos()
        y = r * θ.sin()

        # Convert to pixel grid
        x, y = grid_base.to_pixel_grid(
            grid  = [x,y],
            shape = x.shape[-2:]
        )

        # Remove negative values
        x = x.squeeze(0) - x.min()
        y = y.squeeze(0) - y.min()

    else:
        if shape is None:
            raise Exception("Must pass in either θ/r or shape parameter")

        x = torch.arange(shape[0], device=device)
        y = torch.arange(shape[1], device=device)
        x, y = torch.meshgrid([x,y])

    # Rescale the values based on the specified resolution
    x = x / (x.size(-2) * resolution[0])
    y = y / (y.size(-1) * resolution[1])

    # Determine the number of grid points to generate and modulate any
    # values outside the grid
    xpts = min(x.max().long()+1, repeat)
    ypts = min(y.max().long()+1, repeat)
    if xpts == repeat:
        x = x % repeat
    if ypts == repeat:
        y = y % repeat

    # Calculate the corner grid positions of each pixel
    x_floor = x.floor().long()
    y_floor = y.floor().long()

    x_dist = x - x_floor
    y_dist = y - y_floor
    
    # Generate the gradients
    if corner_grad:
        angles = lambda h,w: torch.where(torch.randn((h,w), device=device) < 0, -1, 1)
        grad_x = angles(xpts+1, ypts+1)
        grad_y = angles(xpts+1, ypts+1)
    else:
        angles = 2 * math.pi * torch.rand(xpts+1, ypts+1, device=device)
        grad_x = torch.cos(angles)
        grad_y = torch.sin(angles)
    
    # Calalate the dot products
    dot = lambda dx, dy : (x_dist - dx)*grad_x[x_floor + dx, y_floor + dy] \
                        + (y_dist - dy)*grad_y[x_floor + dx, y_floor + dy]

    n00 = dot(0,0)
    n01 = dot(0,1)
    n10 = dot(1,0)
    n11 = dot(1,1)

    # Interpolate
    u = fade(x_dist)
    v = fade(y_dist)

    x1 = torch.lerp(n00, n10, u)
    x2 = torch.lerp(n01, n11, u)
    yf = torch.lerp(x1, x2, v)

    if transform is not None:
        yf = transform(yf)

    return yf

def perlin2d_octaves(
    shape       : Tuple[int,int],
    resolution  : Tuple[float,float],
    octaves     : int = 1,
    lacunarity  : float = 2,
    persistence : float = 0.5,
    transform   : Optional[callable] = None,
    device      : Optional[torch.device] = None,
    **kwargs
):
    """
    

    Parameters
    ----------
    shape : Tuple[int,int]
        The shape of the image.

    resolution : tuple[float,float]
        The relative resolution compared to the image size. Features
        will have have a size comparable to the this fractional value
        of the image size.

    octaves : int
        The number of noise sets to generate.

    lacunarity : float
        The factor by which to increase the frequency with each octave.
    
    peristence : float
        The factor by which to decrease the amplitude with each octave.

    transform : callable, optional
        Transformation operation to apply to the noise map output. Only
        used after all the octaves have been applied.

    device : torch.device, optional
        The device to generate data on. Only used if θ/r are not passed.
        Default is None.

    Returns
    -------
    noise : tensor
        The perlin noise map. Values will be between -1 and +1 before
        any transformation function is applied.

    Examples
    --------
    import math
    import matplotlib.pyplot as plt
    from galkit.spatial import coordinate, grid
    from galsyn.random.perlin import perlin2d_octaves

    shape = (128,128)
    grid_base = grid.PytorchGrid()
    grid_kwargs = {
        'h0': 0.1,
        'w0': 0.1,
        'pa': math.pi/4,
        'q' : 0.5,
    }

    θ, r = coordinate.polar(
        grid = grid_base(shape),
        **grid_kwargs
    )

    θ = θ + r.add(1e-6).log().div(math.tan(math.pi/6))

    noise = perlin2d_octaves(θ=θ, r=r,
        grid_base = grid_base,
        grid_kwargs = grid_kwargs,
        resolution = (0.1, 0.1),
        octaves = 3,
        shape = shape,
        transform = lambda x: x.add(1).mul(0.5).pow(2)
    )

    fig, ax = plt.subplots()
    ax.imshow(noise)
    ax.contour(r.squeeze())
    fig.show()    
    """
    noise = torch.zeros(shape, device=device)

    frequency = 1
    amplitude = 1
    max_amplitude = 0
    for _ in range(octaves):
        noise += amplitude * perlin2d(
            shape = shape,
            resolution = [x/frequency for x in resolution],
            device = device,
            **kwargs
        )

        max_amplitude += amplitude
        frequency *= lacunarity
        amplitude *= persistence

    noise /= max_amplitude

    if transform is not None:
        noise = transform(noise)
    
    return noise
