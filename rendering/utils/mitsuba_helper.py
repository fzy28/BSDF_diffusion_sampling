import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb')


def prepare(
    sensor: mi.Sensor,
    seed: int = 0,
    spp: int = 0,
    aovs: list = []):
    """
    Given a sensor and a desired number of samples per pixel, this function
    computes the necessary number of Monte Carlo samples and then suitably
    seeds the sampler underlying the sensor.
    Returns the created sampler and the final number of samples per pixel
    (which may differ from the requested amount depending on the type of
    ``Sampler`` being used)
    Parameter ``sensor`` (``int``, ``mi.Sensor``):
        Specify a sensor to render the scene from a different viewpoint.
    Parameter ``seed` (``int``)
        This parameter controls the initialization of the random number
        generator during the primal rendering step. It is crucial that you
        specify different seeds (e.g., an increasing sequence) if subsequent
        calls should produce statistically independent images (e.g. to
        de-correlate gradient-based optimization steps).
    Parameter ``spp`` (``int``):
        Optional parameter to override the number of samples per pixel for the
        primal rendering step. The value provided within the original scene
        specification takes precedence if ``spp=0``.
    """

    film = sensor.film()
    sampler = sensor.sampler().clone()

    if spp != 0:
        sampler.set_sample_count(spp)

    spp = sampler.sample_count()
    sampler.set_samples_per_wavefront(spp)

    film_size = film.crop_size()

    if film.sample_border():
        film_size += 2 * film.rfilter().border_size()

    wavefront_size = dr.prod(film_size) * spp

    if wavefront_size > 2**32:
        raise Exception(
            "The total number of Monte Carlo samples required by this "
            "rendering task (%i) exceeds 2^32 = 4294967296. Please use "
            "fewer samples per pixel or render using multiple passes."
            % wavefront_size)

    sampler.seed(seed, wavefront_size)
    film.prepare(aovs)

    return sampler, spp

def sample_rays(
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler,
    ):
    """
    Sample a 2D grid of primary rays for a given sensor
    Returns a tuple containing
    - the set of sampled rays
    - a ray weight (usually 1 if the sensor's response function is sampled
      perfectly)
    - the continuous 2D image-space positions associated with each ray
    When a reparameterization function is provided via the 'reparam'
    argument, it will be applied to the returned image-space position (i.e.
    the sample positions will be moving). The other two return values
    remain detached.
    """

    film = sensor.film()
    film_size = film.crop_size()
    rfilter = film.rfilter()
    border_size = rfilter.border_size()

    if film.sample_border():
        film_size += 2 * border_size

    spp = sampler.sample_count()

    # Compute discrete sample position
    idx = dr.arange(mi.UInt32, dr.prod(film_size) * spp)

    # Try to avoid a division by an unknown constant if we can help it
    '''
    log_spp = dr.log2i(spp)
    if 1 << log_spp == spp:
        idx >>= dr.opaque(mi.UInt32, log_spp)
    else:
        idx //= dr.opaque(mi.UInt32, spp)
    '''
        
    # Compute the position on the image plane
    pos = mi.Vector2i()
    pos.y = idx // film_size[0]
    pos.x = dr.fma(-film_size[0], pos.y, idx)

    if film.sample_border():
        pos -= border_size

    pos += mi.Vector2i(film.crop_offset())

    # Cast to floating point and add random offset
    pos_f = mi.Vector2f(pos) + sampler.next_2d()

    # Re-scale the position to [0, 1]^2
    scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
    offset = -mi.ScalarVector2f(film.crop_offset()) * scale
    pos_adjusted = dr.fma(pos_f, scale, offset)

    aperture_sample = mi.Vector2f(0.0)

    time = sensor.shutter_open()

    wavelength_sample = 0

   
    ray, weight = sensor.sample_ray_differential(
        time=time,
        sample1=wavelength_sample,
        sample2=pos_adjusted,
        sample3=aperture_sample
    )


    # With box filter, ignore random offset to prevent numerical instabilities
    splatting_pos = mi.Vector2f(pos) if rfilter.is_box_filter() else pos_f

    return ray, splatting_pos


def mis_weight(pdf_a, pdf_b):
    """
    Compute the Multiple Importance Sampling (MIS) weight given the densities
    of two sampling strategies according to the power heuristic.
    """
    a2 = dr.sqr(pdf_a)
    return dr.detach(dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0), True)