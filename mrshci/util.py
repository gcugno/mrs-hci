import numpy as np
import math
import copy
from astropy.io import fits
from matplotlib import pyplot as plt
from typing import Optional, Tuple, Union

from scipy.ndimage import shift
from astropy.modeling import models, fitting
from sklearn.decomposition import PCA
import pandas as pd

from scipy.ndimage import fourier_shift, shift, rotate
from scipy.stats import t
from scipy.ndimage import gaussian_filter
from skimage.feature import hessian_matrix

from skimage.transform import rescale

from photutils.aperture import aperture_photometry, CircularAperture


def prep_dict_cube(folder, name):
    mrs_data = {}
    mrs_data["1A"] = fits.open(folder+"Level3_ch1-short_s3d.fits")[1].data
    mrs_data["1B"] = fits.open(folder+"Level3_ch1-medium_s3d.fits")[1].data
    mrs_data["1C"] = fits.open(folder+"Level3_ch1-long_s3d.fits")[1].data
    mrs_data["2A"] = fits.open(folder+"Level3_ch2-short_s3d.fits")[1].data
    mrs_data["2B"] = fits.open(folder+"Level3_ch2-medium_s3d.fits")[1].data
    mrs_data["2C"] = fits.open(folder+"Level3_ch2-long_s3d.fits")[1].data
    mrs_data["3A"] = fits.open(folder+"Level3_ch3-short_s3d.fits")[1].data
    mrs_data["3B"] = fits.open(folder+"Level3_ch3-medium_s3d.fits")[1].data
    mrs_data["3C"] = fits.open(folder+"Level3_ch3-long_s3d.fits")[1].data
    mrs_data["name"] = name

    return mrs_data


def prep_wvl_cube(folder):
    mrs_wvl = {}
    mrs_hdr = {}
    hdr = fits.open(folder+"Level3_ch1-short_s3d.fits")[1].header
    mrs_wvl["1A"] = (np.arange(hdr["NAXIS3"])+hdr["CRPIX3"]-1)*hdr["CDELT3"]+hdr["CRVAL3"]
    mrs_hdr["1A"] = hdr
    hdr = fits.open(folder+"Level3_ch1-medium_s3d.fits")[1].header
    mrs_wvl["1B"] = (np.arange(hdr["NAXIS3"])+hdr["CRPIX3"]-1)*hdr["CDELT3"]+hdr["CRVAL3"]
    mrs_hdr["1B"] = hdr
    hdr = fits.open(folder+"Level3_ch1-long_s3d.fits")[1].header
    mrs_wvl["1C"] = (np.arange(hdr["NAXIS3"])+hdr["CRPIX3"]-1)*hdr["CDELT3"]+hdr["CRVAL3"]
    mrs_hdr["1C"] = hdr
    hdr = fits.open(folder+"Level3_ch2-short_s3d.fits")[1].header
    mrs_wvl["2A"] = (np.arange(hdr["NAXIS3"])+hdr["CRPIX3"]-1)*hdr["CDELT3"]+hdr["CRVAL3"]
    mrs_hdr["2A"] = hdr
    hdr = fits.open(folder+"Level3_ch2-medium_s3d.fits")[1].header
    mrs_wvl["2B"] = (np.arange(hdr["NAXIS3"])+hdr["CRPIX3"]-1)*hdr["CDELT3"]+hdr["CRVAL3"]
    mrs_hdr["2B"] = hdr
    hdr = fits.open(folder+"Level3_ch2-long_s3d.fits")[1].header
    mrs_wvl["2C"] = ((np.arange(hdr["NAXIS3"])+hdr["CRPIX3"]-1)*hdr["CDELT3"]+hdr["CRVAL3"])
    mrs_hdr["2C"] = hdr
    hdr = fits.open(folder+"Level3_ch3-short_s3d.fits")[1].header
    mrs_wvl["3A"] = (np.arange(hdr["NAXIS3"])+hdr["CRPIX3"]-1)*hdr["CDELT3"]+hdr["CRVAL3"]
    mrs_hdr["3A"] = hdr
    hdr = fits.open(folder+"Level3_ch3-medium_s3d.fits")[1].header
    mrs_wvl["3B"] = (np.arange(hdr["NAXIS3"])+hdr["CRPIX3"]-1)*hdr["CDELT3"]+hdr["CRVAL3"]
    mrs_hdr["3B"] = hdr
    hdr = fits.open(folder+"Level3_ch3-long_s3d.fits")[1].header
    mrs_wvl["3C"] = (np.arange(hdr["NAXIS3"])+hdr["CRPIX3"]-1)*hdr["CDELT3"]+hdr["CRVAL3"]
    mrs_hdr["3C"] = hdr
    return mrs_wvl, mrs_hdr


def find_star(frame):
    frame[np.isnan(frame)] = 0.
    y, x = np.mgrid[:frame.shape[0], :frame.shape[1]]
    y0, x0 = x.flatten()[np.argmax(frame)], y.flatten()[np.argmax(frame)]
    # fit center
    pinit = models.Gaussian2D(np.max(frame), y0, x0, 1.0, 1.0 )
    fitp = fitting.LevMarLSQFitter()
    params = fitp(pinit, x, y, frame)
    y0, x0 = params.x_mean.value, params.y_mean.value
    return x0, y0

    
def apply_PCA(pca_number,
             s,
             rm):
             
    r = copy.copy(rm)
    # data size
    im_shape = s.shape
    
    # prepare residuals
    residuals = np.zeros_like(s)
    
    # vectorize science and refs
    science_vec = s.reshape(1,im_shape[0]*im_shape[1])
    refs_vec = r.reshape(len(r), im_shape[0]*im_shape[1])
    
    mean_ref = np.mean(refs_vec, axis=0)
    refs_vec -= mean_ref
    
    # find components
    pca_sklearn = PCA(n_components=pca_number, svd_solver='arpack')
    pca_sklearn.fit(refs_vec)
    
    # Add mean of the refs as first component
    mean_ref_reshape = mean_ref.reshape((1, mean_ref.shape[0]))
    q_ortho, _ = np.linalg.qr(np.vstack((mean_ref_reshape, pca_sklearn.components_[:-1, ])).T)
    pca_sklearn.components_ = q_ortho.T
    
    # find psf model and residuals
    pca_rep = np.matmul(pca_sklearn.components_[:pca_number], science_vec.T)
    zeros = np.zeros((pca_sklearn.n_components - pca_number, science_vec.shape[0]))
    pca_rep = np.vstack((pca_rep, zeros)).T
    psf_model_vec = pca_sklearn.inverse_transform(pca_rep)
    psf_model = psf_model_vec.reshape((im_shape[0],im_shape[1]))
    residuals = (science_vec - psf_model_vec).reshape((im_shape[0],im_shape[1]))

    return psf_model, residuals




def crop_science(data, data_hdr, band, out_dir, size):
    # Take data of the right band, remove nans and collapse to 2D
    dd = data#[band]
    dd[np.isnan(dd)] = 0.
    d = np.mean(dd, axis=0)
    # Find the stellar position
    dy0, dx0 = find_star(d)
    # Cut the science cube around that position
    science = dd[:,round(dy0)-size:round(dy0)+size+1, round(dx0)-size:round(dx0)+size+1]*(data_hdr["PIXAR_SR"]*1e6)
        
    # Save the offset for later
    pt2 = np.vstack((dy0-round(dy0), dx0-round(dx0)))
    pd_df2 = pd.DataFrame(pt2.T)
    pd_df2.to_csv(out_dir + f'/Image_shifts/Shift_{band}.txt', index=False, header=('y [pix]', 'x [pix]'), sep='\t')
    return science, (dx0,dy0)

def crop_refs(refs_list, refs_names, data_hdr, band, D0, out_dir, size):
    dx0,dy0 = D0
    refs = []
    # Loop over the references
    for i, ref in enumerate(refs_list):
        # Take data of the right band, remove nans and collapse to 2D
        r = ref
        r[np.isnan(r)] = 0.
        rm= np.mean(r, axis=0)
        # Find the stellar position
        ry0, rx0 = find_star(rm)
        # Find the shift to align with science data
        center_shift = [dy0-ry0, dx0-rx0]
        # Shift the references to match the star location and crop
        ref_i_centered = []
        for k in range(len(r)):
            rshift = shift(r[k], (center_shift[0], center_shift[1]), order=5)
            ref_i_centered.append(rshift[round(dy0)-size:round(dy0)+size+1, round(dx0)-size:round(dx0)+size+1])
        refs.append(ref_i_centered)
    # Save cropped references in a new dictionary
    refs_dict = np.array(refs)*(data_hdr["PIXAR_SR"]*1e6)

    # Plot the reference positions
    fig, ax = plt.subplots(ncols=6, nrows=3, figsize=(10, 6))
    ax = np.array(ax).flatten()

    #for i, ref in enumerate(refs_list):
    i=0
    v = np.max(np.nanmean(refs_dict[0], axis=0))
    while i<len(refs_names):
        val = 2e-2
        im = np.nanmean(refs_dict[i], axis=0)
        ax[i].imshow(im, origin='lower', vmin=-v, vmax=v, cmap='RdBu_r')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].text(0, 2, refs_names[i], size=10, color="black")
        i += 1

    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.05, hspace=None)
    fig.savefig(out_dir + f'/Img/Refs_{band}.pdf')
        
    return refs_dict


def crop_psf(psf_cube, data_hdr, band, out_dir, size):

    p = psf_cube
    p[np.isnan(p)] = 0.
    pm= np.mean(p, axis=0)
    py0, px0 = find_star(pm)
    pic_cen = (np.shape(pm)[0]/2,np.shape(pm)[1]/2) #- (0.5,0.5)
    center_shift = [pic_cen[0]-py0-0.5, pic_cen[1]-px0-0.5]
    psf_i_centered = []
    diff_y = int((np.shape(pm)[1]-(2*size+1))/2)
    diff_x = int((np.shape(pm)[0]-(2*size+1))/2)
    for k in range(len(p)):
        pshift = shift(p[k], (center_shift[0], center_shift[1]), order=5)
        psf_i_centered.append(pshift[diff_x:-diff_x, diff_y:-diff_y])
    psf = np.array(psf_i_centered)*(data_hdr["PIXAR_SR"]*1e6)
    
    val = 2e-2
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(9, 4))
    ax.imshow(np.nanmean(psf, axis=0)[7:-7,7:-7], origin='lower', vmin=-val, vmax=val, cmap='RdBu_r')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0, 2, "PSF", size=10, color="black")
    fig.savefig(out_dir + f'/Img/PSF_{band}.pdf')
    return psf


def _objective(arg,
               sep_ang,
               image,
               refs,
               psf_im,
               ap_pix_k,
               pca_number,
               var_noise,
               mask,
               verb=False,
              ):
    mag=arg[0]

    ap_pos = polar_to_cartesian(image, sep = sep_ang[0], ang = sep_ang[1])
    aperture = (round(ap_pos[0]), round(ap_pos[1]), ap_pix_k)
    
    # Inject the negative artifical planet at the position and contrast that is tested
    fake = fake_planet(images=image,
                       psf=psf_im,
                       parang=np.array([0]),
                       position=(sep_ang[0], sep_ang[1]),
                       magnitude=mag,
                       psf_scaling=-1)
                                                               
    if pca_number is not None:
        _, res = apply_PCA(pca_number, fake[0]*mask, refs*mask)
    else:
        res = (fake[0]*mask-refs*mask)
    
    # Calculate the chi-square for the tested position and contrast
    chi_sq = merit_function(residuals=res,
                                    merit='poisson',
                                    aperture=aperture,
                                    sigma=0,
                                    var_noise=var_noise)

    return chi_sq





def remove_outliers(wavelength, spectrum, spectrum_err, threshold=3):
    """
    Remove outliers from wavelength and spectrum arrays using the Z-score method.

    Parameters:
    - wavelength: List or NumPy array of wavelength values.
    - spectrum: List or NumPy array of corresponding spectrum values.
    - threshold: Z-score threshold for identifying outliers. Default is 3.

    Returns:
    - Two lists (wavelength and spectrum) with outliers removed.
    """
    wavelength = np.array(wavelength)
    spectrum = np.array(spectrum)
    
    # Calculate the median of the spectrum values
    median_spectrum = np.median(spectrum)
    
    # Calculate the Median Absolute Deviation (MAD)
    mad = np.median(np.abs(spectrum - median_spectrum))

    # Calculate Z-scores for the spectrum values
    z_scores = np.abs((spectrum - median_spectrum) / (mad * 1.4826))

    # Filter both wavelength and spectrum arrays based on the threshold
    filtered_wavelength1 = wavelength[z_scores < threshold]
    filtered_spectrum1 = spectrum[z_scores < threshold]
    filtered_spectrum_err1 = spectrum_err[z_scores < threshold]
    
    
    ### Repeat the same with errorbars
    # Calculate the median of the uncertainties values
    median_spectrum_err = np.median(filtered_spectrum_err1)
    
    # Calculate the Median Absolute Deviation (MAD)
    mad_err = np.median(np.abs(filtered_spectrum_err1 - median_spectrum_err))
    
    # Calculate Z-scores for the spectrum values
    z_scores_err = np.abs((filtered_spectrum_err1 - median_spectrum_err) / (mad_err * 1.4826))
    
    # Filter both wavelength and spectrum arrays based on the threshold
    filtered_wavelength = filtered_wavelength1[z_scores_err < threshold]
    filtered_spectrum = filtered_spectrum1[z_scores_err < threshold]
    filtered_spectrum_err = filtered_spectrum_err1[z_scores_err < threshold]

    return filtered_wavelength, filtered_spectrum, filtered_spectrum_err


# PYNPOINT functions

def create_mask(im_shape: Tuple[int, int],
                size: Union[Tuple[float, float],
                            Tuple[float, None],
                            Tuple[None, float],
                            Tuple[None, None]]) -> np.ndarray:
    """
    Function to create a mask for the central and outer image regions.

    Parameters
    ----------
    im_shape : tuple(int, int)
        Image size in both dimensions.
    size : tuple(float, float)
        Size (pix) of the inner and outer mask.

    Returns
    -------
    np.ndarray
        Image mask.
    """

    mask = np.ones(im_shape)
    npix = im_shape[0]

    if size[0] is not None or size[1] is not None:

        if npix % 2 == 0:
            x_grid = y_grid = np.linspace(-npix / 2 + 0.5, npix / 2 - 0.5, npix)
        else:
            x_grid = y_grid = np.linspace(-(npix - 1) / 2, (npix - 1) / 2, npix)

        xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
        rr_grid = np.sqrt(xx_grid**2 + yy_grid**2)

        if size[0] is not None:
            mask[rr_grid < size[0]] = 0.

        if size[1] is not None:
            if size[1] > npix / 2:
                size = (size[0], npix / 2)
            mask[rr_grid > size[1]] = 0.

    return mask


def center_subpixel(image: np.ndarray) -> Tuple[float, float]:
    """
    Function to get the precise position of the image center. The center of the pixel in the
    bottom left corner of the image is defined as (0, 0), so the bottom left corner of the
    image is located at (-0.5, -0.5).

    Parameters
    ----------
    image : np.ndarray
        Input image (2D or 3D).

    Returns
    -------
    tuple(float, float)
        Subpixel position (y, x) of the image center.
    """

    center_x = float(image.shape[-1]) / 2 - 0.5
    center_y = float(image.shape[-2]) / 2 - 0.5

    return center_y, center_x


def cartesian_to_polar(center: Tuple[float, float],
                       y_pos: float,
                       x_pos: float) -> Tuple[float, float]:
    """
    Function to convert pixel coordinates to polar coordinates.

    Parameters
    ----------
    center : tuple(float, float)
        Image center (y, x) from :func:`~pynpoint.util.image.center_subpixel`.
    y_pos : float
        Pixel coordinate along the vertical axis. The bottom left corner of the image is
        (-0.5, -0.5).
    x_pos : float
        Pixel coordinate along the horizontal axis. The bottom left corner of the image is
        (-0.5, -0.5).

    Returns
    -------
    tuple(float, float)
        Separation (pix) and position angle (deg). The angle is measured counterclockwise with
        respect to the positive y-axis.
    """

    sep = math.sqrt((center[1] - x_pos)**2 + (center[0] - y_pos)**2)
    ang = math.atan2(y_pos-center[1], x_pos-center[0])
    ang = (math.degrees(ang) - 90) % 360

    return sep, ang


def select_annulus(image_in: np.ndarray,
                   radius_in: float,
                   radius_out: float,
                   mask_position: Optional[Tuple[float, float]] = None,
                   mask_radius: Optional[float] = None) -> np.ndarray:
    """
    image_in : np.ndarray
        Input image.
    radius_in : float
        Inner radius of the annulus (pix).
    radius_out : float
        Outer radius of the annulus (pix).
    mask_position : tuple(float, float), None
        Center (pix) position (y, x) in of the circular region that is excluded. Not used
        if set to None.
    mask_radius : float, None
        Radius (pix) of the circular region that is excluded. Not used if set to None.
    """

    im_shape = image_in.shape

    if im_shape[0] % 2 == 0:
        y_grid = np.linspace(-im_shape[0] / 2 + 0.5, im_shape[0] / 2 - 0.5, im_shape[0])
    else:
        y_grid = np.linspace(-(im_shape[0] - 1) / 2, (im_shape[0] - 1) / 2, im_shape[0])

    if im_shape[1] % 2 == 0:
        x_grid = np.linspace(-im_shape[1] / 2 + 0.5, im_shape[1] / 2 - 0.5, im_shape[1])
    else:
        x_grid = np.linspace(-(im_shape[1] - 1) / 2, (im_shape[1] - 1) / 2, im_shape[1])

    xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)
    rr_grid = np.sqrt(xx_grid**2 + yy_grid**2)

    mask = np.ones(im_shape)

    indices = np.where((rr_grid < radius_in) | (rr_grid > radius_out))
    mask[indices[0], indices[1]] = 0.

    if mask_position is not None and mask_radius is not None:
        distance = subpixel_distance(im_shape=im_shape, position=mask_position)
        indices = np.where(distance < mask_radius)
        mask[indices[0], indices[1]] = 0.

    indices = np.where(mask == 1.)

    return image_in[indices[0], indices[1]]


def polar_to_cartesian(image: np.ndarray,
                       sep: float,
                       ang: float) -> Tuple[float, float]:
    """
    Function to convert polar coordinates to pixel coordinates.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D or 3D).
    sep : float
        Separation (pixels).
    ang : float
        Position angle (deg), measured counterclockwise with respect to the positive y-axis.

    Returns
    -------
    tuple(float, float)
        Cartesian coordinates (y, x). The bottom left corner of the image is (-0.5, -0.5).
    """

    center = center_subpixel(image)  # (y, x)

    x_pos = center[1] + sep * math.cos(math.radians(ang + 90))
    y_pos = center[0] + sep * math.sin(math.radians(ang + 90))

    return y_pos, x_pos


def pixel_distance(im_shape: Tuple[int, int],
                   position: Optional[Tuple[int, int]] = None) -> Tuple[
                       np.ndarray, np.ndarray, np.ndarray]:
    """
    Function to calculate the distance of each pixel with respect to a given pixel position.
    Supports both odd and even sized images.

    Parameters
    ----------
    im_shape : tuple(int, int)
        Image shape (y, x).
    position : tuple(int, int)
        Pixel center (y, x) from which the distance is calculated. The image center is used if set
        to None. Python indexing starts at zero so the center of the bottom left pixel is (0, 0).

    Returns
    -------
    np.ndarray
        2D array with the distances of each pixel from the provided pixel position.
    np.ndarray
        2D array with the x coordinates.
    np.ndarray
        2D array with the y coordinates.
    """

    if im_shape[0] % 2 == 0:
        y_grid = np.linspace(-im_shape[0] / 2 + 0.5, im_shape[0] / 2 - 0.5, im_shape[0])

    else:
        y_grid = np.linspace(-(im_shape[0] - 1) / 2, (im_shape[0] - 1) / 2, im_shape[0])

    if im_shape[1] % 2 == 0:
        x_grid = np.linspace(-im_shape[1] / 2 + 0.5, im_shape[1] / 2 - 0.5, im_shape[1])

    else:
        x_grid = np.linspace(-(im_shape[1] - 1) / 2, (im_shape[1] - 1) / 2, im_shape[1])

    if position is not None:
        y_shift = y_grid[position[0]]
        x_shift = x_grid[position[1]]

        y_grid -= y_shift
        x_grid -= x_shift

    xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)

    return np.sqrt(xx_grid**2 + yy_grid**2), xx_grid, yy_grid


def subpixel_distance(im_shape: Tuple[int, int],
                      position: Tuple[float, float],
                      shift_center: bool = True) -> np.ndarray:
    """
    Function to calculate the distance of each pixel with respect to a given subpixel position.
    Supports both odd and even sized images.

    Parameters
    ----------
    im_shape : tuple(int, int)
        Image shape (y, x).
    position : tuple(float, float)
        Pixel center (y, x) from which the distance is calculated. Python indexing starts at zero
        so the bottom left image corner is (-0.5, -0.5).
    shift_center : bool
        Apply the coordinate correction for the image center.

    Returns
    -------
    np.ndarray
        2D array with the distances of each pixel from the provided pixel position.
    """

    # Get 2D x and y coordinates with respect to the image center
    _, xx_grid, yy_grid = pixel_distance(im_shape, position=None)

    if im_shape[0] % 2 == 0:
        # Distance from the image center to the center of the outermost pixel
        # Even sized images
        y_size = im_shape[0] / 2 + 0.5
        x_size = im_shape[1] / 2 + 0.5

    else:
        # Distance from the image center to the center of the outermost pixel
        # Odd sized images
        y_size = (im_shape[0] - 1) / 2
        x_size = (im_shape[1] - 1) / 2

    if shift_center:
        # Shift the image center to the center of the bottom left pixel
        yy_grid += y_size
        xx_grid += x_size

    # Apply a subpixel shift of the coordinate system to the requested position
    yy_grid -= position[0]
    xx_grid -= position[1]

    return np.sqrt(xx_grid**2 + yy_grid**2)

# PYNPOINT: utils.analysis.py


def compute_aperture_flux_elements(image: np.ndarray,
                                   x_pos: float,
                                   y_pos: float,
                                   size: float,
                                   ignore: bool):
    """
    Computes the average fluxes inside apertures with the same separation from the center.
    This function can be used to to estimate the residual flux of a planet at position
    (x_pos, y_pos) and the respective noise elements with same separation (see function false_alarm)
    It can also be used to compute the noise apertures is if no planet is present
    (needed for contrast curves).

    Parameters
    ----------
    image : numpy.ndarray
        The input image as a 2D numpy array. For example, this could be a residual frame returned by
        a :class:`.PcaPsfSubtractionModule`.
    x_pos : float
        The planet position (in pixels) along the horizontal axis. The pixel coordinates of the
        bottom-left corner of the image are (-0.5, -0.5). If no planet is present x_pos and y_pos
        determine the separation from the center.
    y_pos : float
        The planet position (pix) along the vertical axis. The pixel coordinates of the bottom-left
        corner of the image are (-0.5, -0.5). If no planet is present x_pos and y_pos
        determine the separation from the center.
    size : float
        The radius of the reference apertures (in pixels). Usually, this value is chosen close to
        one half of the typical FWHM of the PSF (0.514 lambda over D for a perfect Airy pattern; in
        practice, however, the FWHM is often larger than this).
    ignore : bool
        Whether or not to ignore the immediate neighboring apertures for the noise estimate. This is
        desirable in case there are "self-subtraction wings" left and right of the planet which
        would bias the estimation of the noise level at the separation of the planet if not ignored.

    Returns
    -------
    ap_phot :
        A list of aperture photometry values. If a planet was present ap_phot[0] contains the flux
        of the planet and ap_phot[1:] contains the noise. If not planet was present ap_phot[...]
        gives the aperture photometry of the noise elements.
    """

    # Compute the center of the current frame (with subpixel precision) and use it to compute the
    # radius of the given position in polar coordinates (with the origin at the center of the frame)
    center = center_subpixel(image)
    radius = math.sqrt((center[0] - y_pos)**2 + (center[1] - x_pos)**2)

    # Compute the number of apertures which we can place at the separation of  the given position
    num_ap = int(math.pi * radius / size)

    # Compute the angles at which to place the reference apertures
    ap_theta = np.linspace(0, 2 * math.pi, num_ap, endpoint=False)

    # If ignore is True, delete the apertures immediately right and left of the aperture placed on
    # the planet signal. These apertures often contain "self-subtraction wings", which means they
    # cannot be considered to originate from the same distribution. In accordance with section 3.2
    # of Mawet et al. (2014), such apertures are ignored to prevent bias.
    if ignore:
        num_ap -= 2
        ap_theta = np.delete(ap_theta, [1, np.size(ap_theta) - 1])

    # If the number of apertures is 2 or less, we cannot compute the false positive fraction
    if num_ap < 3:
        raise ValueError(
            f'Number of apertures (num_ap={num_ap}) is too small to calculate the '
            'false positive fraction.')

    # Initialize a numpy array in which we will store the integrated flux of all reference apertures
    ap_phot = np.zeros(num_ap)

    # Loop over all reference apertures and measure the integrated flux
    for i, theta in enumerate(ap_theta):
        # Compute the position of the current aperture in polar coordinates and convert to Cartesian
        x_tmp = center[1] + (x_pos - center[1]) * math.cos(theta) - \
            (y_pos - center[0]) * math.sin(theta)
        y_tmp = center[0] + (x_pos - center[1]) * math.sin(theta) + \
            (y_pos - center[0]) * math.cos(theta)

        # Place a circular aperture at a position and sum up the flux inside the aperture
        aperture = CircularAperture((x_tmp, y_tmp), size)
        phot_table = aperture_photometry(image, aperture, method='exact')

        ap_phot[i] = phot_table['aperture_sum']

    return ap_phot


def false_alarm(image: np.ndarray,
                x_pos: float,
                y_pos: float,
                size: float,
                ignore: bool) -> Tuple[float, float, float, float]:
    """
    Compute the signal-to-noise ratio (SNR), which is formally defined as the test statistic of a
    two-sample t-test, and related quantities (such as the FPF) at a given position in an image.

    For more detailed information about the definition of the signal-to-noise ratio and the
    motivation behind it, please see the following paper:

        Mawet, D. et al. (2014): "Fundamental limitations of high contrast imaging set by small
        sample statistics". *The Astrophysical Journal*, 792(2), 97.
        DOI: `10.1088/0004-637X/792/2/97 <https://dx.doi.org/10.1088/0004-637X/792/2/97>`_.

    Parameters
    ----------
    image : numpy.ndarray
        The input image as a 2D numpy array. For example, this could be a residual frame returned by
        a :class:`.PcaPsfSubtractionModule`.
    x_pos : float
        The planet position (in pixels) along the horizontal axis. The pixel coordinates of the
        bottom-left corner of the image are (-0.5, -0.5).
    y_pos : float
        The planet position (pix) along the vertical axis. The pixel coordinates of the bottom-left
        corner of the image are (-0.5, -0.5).
    size : float
        The radius of the reference apertures (in pixels). Usually, this value is chosen close to
        one half of the typical FWHM of the PSF (0.514 lambda over D for a perfect Airy pattern; in
        practice, however, the FWHM is often larger than this).
    ignore : bool
        Whether or not to ignore the immediate neighboring apertures for the noise estimate. This is
        desirable in case there are "self-subtraction wings" left and right of the planet which
        would bias the estimation of the noise level at the separation of the planet if not ignored.

    Returns
    -------
    signal_sum :
        The integrated (summed up) flux inside the signal aperture.

        Please note that this is **not** identical to the numerator of the fraction defining the SNR
        (which is given by the `signal_sum` minus the mean of the noise apertures).
    noise :
        The denominator of the SNR, i.e., the standard deviation of the integrated flux of the noise
        apertures, times a correction factor that accounts for small sample statistics.
    snr :
        The signal-to-noise ratio (SNR) as defined by Mawet et al. (2014) in eq. (8).
    fpf :
        The false positive fraction (FPF) as defined by Mawet et al. (2014) in eq. (10).
    """

    ap_phot = compute_aperture_flux_elements(image=image,
                                             x_pos=x_pos,
                                             y_pos=y_pos,
                                             size=size,
                                             ignore=ignore)

    # Define shortcuts to the signal and the noise aperture sums
    signal_aperture = ap_phot[0]
    noise_apertures = ap_phot[1:]

    # Compute the "signal", that is, the numerator of the signal-to-noise ratio: According to
    # eq. (8) in Mawet et al. (2014), this is given by the difference between the integrated flux
    # in the signal aperture and the mean of the integrated flux in the noise apertures
    signal = signal_aperture - np.mean(noise_apertures)

    # Compute the "noise", that is, the denominator of the signal-to-noise-ratio: According to
    # eq. (8) in Mawet et al. (2014), this is given by the standard deviation of the integrated flux
    # in the noise apertures times a correction factor to account for the small sample statistics.
    # NOTE: `ddof=1` is a necessary argument for np.std() in order to compute the *unbiased*
    #       estimate (i.e., including Bessel's corrections) of the standard deviation.
    noise = np.std(noise_apertures, ddof=1) *\
        math.sqrt(1 + 1 / (noise_apertures.shape[0]))

    # Compute the signal-to-noise ratio by dividing the "signal" through the "noise"
    snr = signal / noise

    # Compute the false positive fraction (FPF). According to eq. (10) in Mawet et al. (2014), the
    # FPF is given by 1 - F_nu(SNR), where F_nu is the cumulative distribution function (CDF) of a
    # t-distribution with `nu = n-1` degrees of freedom (see Section 3 of Mawet et al. (2014) for
    # more details on the Student's t distribution).
    # For numerical reasons, we use the survival function (SF), which is defined precisely as 1-CDF,
    # but may give more accurate results according to the scipy documentation.
    fpf = t.sf(snr, df=(noise_apertures.shape[0] - 1))

    return signal_aperture, noise, snr, fpf


def shift_image(image: np.ndarray,
                shift_yx: Union[Tuple[float, float], np.ndarray],
                interpolation: str,
                mode: str = 'constant') -> np.ndarray:
    """
    Function to shift an image.

    Parameters
    ----------
    image : np.ndarray
        Input image (2D or 3D). If 3D the image is not shifted along the 0th axis.
    shift_yx : tuple(float, float), np.ndarray
        Shift (y, x) to be applied (pix). An additional shift of zero pixels will be added
        for the first dimension in case the input image is 3D.
    interpolation : str
        Interpolation type ('spline', 'bilinear', or 'fft').
    mode : str
        Interpolation mode.

    Returns
    -------
    np.ndarray
        Shifted image.
    """

    if image.ndim == 2:
        shift_val = (shift_yx[0], shift_yx[1])
    elif image.ndim == 3:
        shift_val = (0, shift_yx[0], shift_yx[1])
    else:
        raise ValueError('Invalid number of dimensions for image: must be 2 or 3')

    if interpolation == 'spline':
        im_center = shift(image, shift_val, order=5, mode=mode)

    elif interpolation == 'bilinear':
        im_center = shift(image, shift_val, order=1, mode=mode)

    elif interpolation == 'fft':
        fft_shift = fourier_shift(np.fft.fftn(image), shift_val)
        im_center = np.fft.ifftn(fft_shift).real

    else:
        raise ValueError('interpolation must be one of the following: spline, bilinear, fft')

    return im_center


def fake_planet(images: np.ndarray,
                psf: np.ndarray,
                parang: np.ndarray,
                position: Tuple[float, float],
                magnitude: float,
                psf_scaling: float,
                interpolation: str = 'spline') -> np.ndarray:
    """
    Function to inject artificial planets in a dataset.

    Parameters
    ----------
    images : numpy.ndarray
        Input images (3D).
    psf : numpy.ndarray
        PSF template (3D).
    parang : numpy.ndarray
        Parallactic angles (deg).
    position : tuple(float, float)
        Separation (pix) and position angle (deg) measured in counterclockwise with respect to the
        upward direction.
    magnitude : float
        Magnitude difference used to scale input PSF.
    psf_scaling : float
        Extra factor used to scale input PSF.
    interpolation : str
        Interpolation type ('spline', 'bilinear', or 'fft').

    Returns
    -------
    numpy.ndarray
        Images with artificial planet injected.
    """

    sep = position[0]
    ang = np.radians(position[1] + 90. - parang)

    flux_ratio = 10. ** (-magnitude / 2.5)
    psf = psf*psf_scaling*flux_ratio

    x_shift = sep*np.cos(ang)
    y_shift = sep*np.sin(ang)

    im_shift = np.zeros(images.shape)

    for i in range(images.shape[0]):
        if psf.shape[0] == 1:
            im_shift[i, ] = shift_image(psf[0, ],
                                        (float(y_shift[i]), float(x_shift[i])),
                                        interpolation,
                                        mode='reflect')

        else:
            im_shift[i, ] = shift_image(psf[i, ],
                                        (float(y_shift[i]), float(x_shift[i])),
                                        interpolation,
                                        mode='reflect')

    return images + im_shift


def merit_function(residuals: np.ndarray,
                   merit: str,
                   aperture: Tuple[int, int, float],
                   sigma: float,
                   var_noise: Optional[float]) -> float:
    """
    Function to calculate the figure of merit at a given position in the image residuals.

    Parameters
    ----------
    residuals : numpy.ndarray
        Residuals of the PSF subtraction (2D).
    merit : str
        Figure of merit for the chi-square function ('hessian', 'poisson', or 'gaussian').
    aperture : tuple(int, int, float)
        Position (y, x) of the aperture center (pix) and aperture radius (pix).
    sigma : float
        Standard deviation (pix) of the Gaussian kernel which is used to smooth the residuals
        before the chi-square is calculated.
    var_noise : float, None
        Variance of the noise which is required when `merit` is set to 'gaussian' or 'hessian'.

    Returns
    -------
    float
        Chi-square value.
    """

    rr_grid, _, _ = pixel_distance(residuals.shape, position=(aperture[0], aperture[1]))

    indices = np.where(rr_grid <= aperture[2])

    if merit == 'hessian':

        hessian_rr, hessian_rc, hessian_cc = hessian_matrix(image=residuals,
                                                            sigma=sigma,
                                                            mode='constant',
                                                            cval=0.,
                                                            order='rc',
                                                            use_gaussian_derivatives=False)

        hes_det = (hessian_rr*hessian_cc) - (hessian_rc*hessian_rc)

        chi_square = np.sum(hes_det[indices]**2)/var_noise

    elif merit == 'poisson':

        if sigma > 0.:
            residuals = gaussian_filter(input=residuals, sigma=sigma)

        chi_square = np.sum(np.abs(residuals[indices]))

    elif merit == 'gaussian':

        chi_square = np.sum(residuals[indices]**2)/var_noise

    else:

        raise ValueError('Figure of merit not recognized. Please use \'hessian\', \'poisson\' '
                         'or \'gaussian\'. Previous use of \'sum\' should now be set as '
                         '\'poisson\'.')

    return chi_square

def pa_to_MRS_pa(pa, v3pa, band):
    mrs_rot = {"1": 8.3, "2": 8.2, "3": 7.6, "4": 8.4}
    north = -(v3pa + 180 + mrs_rot[band[0]]) # the 180 comes from the orientation of alpha/beta with respect to V3
    return north + pa

