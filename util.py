import numpy as np
import copy
from astropy.io import fits
from matplotlib import pyplot as plt

from scipy.ndimage import shift
from astropy.modeling import models, fitting
from sklearn.decomposition import PCA
import pandas as pd

import sys
sys.path.append('/Users/gcugno/Tools/PynPoint')
from pynpoint.util.analysis import fake_planet, merit_function
from pynpoint.util.image import polar_to_cartesian

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
    filtered_wavelength = wavelength[z_scores < threshold]
    filtered_spectrum = spectrum[z_scores < threshold]
    filtered_spectrum_err = spectrum_err[z_scores < threshold]

    return filtered_wavelength, filtered_spectrum, filtered_spectrum_err
