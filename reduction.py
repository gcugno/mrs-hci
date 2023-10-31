import os
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from tqdm import tqdm


from util import prep_dict_cube, prep_wvl_cube, find_star, apply_PCA, crop_science, crop_refs, crop_psf, _objective
from fixed_values import *

import sys
sys.path.append('/Users/gcugno/Tools/PynPoint')
from pynpoint.util.image import create_mask, polar_to_cartesian, select_annulus, cartesian_to_polar
from pynpoint.util.analysis import false_alarm, fake_planet


class MRS_HCI:
    def __init__(self,
                 output_dir,
                 science_path,
                 band = "1A",
                 verbose = True,# pixels
                ):

        self.output_dir = output_dir
        self.band = band
        self.verbose = verbose
                
        
        if not os.path.isdir(output_dir+'/Image_shifts'):
            os.mkdir(output_dir+'/Image_shifts')
        if not os.path.isdir(output_dir+'/Img'):
            os.mkdir(output_dir+'/Img')
        if not os.path.isdir(output_dir+'/Manipulated_images'):
            os.mkdir(output_dir+'/Manipulated_images')
        if not os.path.isdir(output_dir+'/Residuals'):
            os.mkdir(output_dir+'/Residuals')
            
            
        self.data_import = prep_dict_cube(science_path, 'SCIENCE')[self.band]
        self.wvl, self.data_hdr = prep_wvl_cube(science_path)
        self.wvl = self.wvl[self.band]
        self.data_hdr = self.data_hdr[self.band]
        
        self.pixelsize= self.data_hdr["CDELT1"]*3600
        
                #####################################################
        ## THIS SHOULD BE automated!!
        add = 'MRS_REF_PSF_processed_270823/'
        ref_1538_1 = prep_dict_cube("/Users/gcugno/Science/JWST/MRS/GQLup/GQLup_MIRI/"+add+"1538/cubes_obs1/", 'PID1538_obs1')[self.band]
        ref_1538_2 = prep_dict_cube("/Users/gcugno/Science/JWST/MRS/GQLup/GQLup_MIRI/"+add+"1538/cubes_obs2/", 'PID1538_obs2')[self.band]
        ref_1538_3 = prep_dict_cube("/Users/gcugno/Science/JWST/MRS/GQLup/GQLup_MIRI/"+add+"1538/cubes_obs3/", 'PID1538_obs3')[self.band]
        ref_1536_22 = prep_dict_cube("/Users/gcugno/Science/JWST/MRS/GQLup/GQLup_MIRI/"+add+"1536/cubes_obs22/", 'PID1536_obs22')[self.band]
        ref_1536_23 = prep_dict_cube("/Users/gcugno/Science/JWST/MRS/GQLup/GQLup_MIRI/"+add+"1536/cubes_obs23/", 'PID1536_obs23')[self.band]
        ref_1536_24 = prep_dict_cube("/Users/gcugno/Science/JWST/MRS/GQLup/GQLup_MIRI/"+add+"1536/cubes_obs24/", 'PID1536_obs24')[self.band]
        ref_1524_17 = prep_dict_cube("/Users/gcugno/Science/JWST/MRS/GQLup/GQLup_MIRI/"+add+"1524/cubes_obs17/", 'PID1524_obs17')[self.band]
        ref_1050_3 = prep_dict_cube("/Users/gcugno/Science/JWST/MRS/GQLup/GQLup_MIRI/"+add+"1050/cubes_obs3/", 'PID1050_obs3')[self.band]
        ref_1050_9 = prep_dict_cube("/Users/gcugno/Science/JWST/MRS/GQLup/GQLup_MIRI/"+add+"1050/cubes_obs9/", 'PID1050_obs9')[self.band]
        ref_RYLup = prep_dict_cube("/Users/gcugno/Science/JWST/MRS/GQLup/GQLup_MIRI/VRYLUP/cubes/", 'RYLup')[self.band]
        
        self.refs_list = [ref_1538_1, ref_1538_2, ref_1538_3, ref_1536_22, ref_1536_23, ref_1536_24, ref_1524_17, ref_1050_3, ref_1050_9, ref_RYLup]
        
        self.psf_import = prep_dict_cube("/Users/gcugno/Science/JWST/MRS/GQLup/GQLup_MIRI/"+add+"1536/cubes_obs22/", 'PSF')[self.band]
        
        print ('[WARNING]\t[The path to the references is hardcoded]')
        #####################################################
        
        return
    
    
    
    def prepare_cubes(self,
                     size = 10):
        
        self.size = size
        
        ## CROPPING the cubes
        self.science, D0 = crop_science(self.data_import, self.data_hdr, self.band, self.output_dir, size)
        self.refs_dict = crop_refs(self.refs_list, self.data_hdr, self.band, D0, self.output_dir, size)
        self.psf = crop_psf(self.psf_import, self.data_hdr, self.band, self.output_dir, size)
        if self.verbose:
            print ('[DONE]\t\t[All the data have been cropped]')

            
        #for band in self.bands:
        hdul = fits.HDUList(fits.PrimaryHDU())
        hdul.append(fits.ImageHDU(self.refs_dict, name="REF"))
        hdul.writeto(self.output_dir + f'/Manipulated_images/References_{self.band}.fits', overwrite=True)

        hdul = fits.HDUList(fits.PrimaryHDU())
        hdul.append(fits.ImageHDU(self.science, name="SCI"))
        hdul.writeto(self.output_dir + f'/Manipulated_images/Science_{self.band}.fits', overwrite=True)

        hdul = fits.HDUList(fits.PrimaryHDU())
        hdul.append(fits.ImageHDU(self.psf, name="PSF"))
        hdul.writeto(self.output_dir + f'/Manipulated_images/PSF_{self.band}.fits', overwrite=True)
        if self.verbose:
            print ('[DONE]\t\t[All the data saved in fits files]')
        #####################################################
        #INCLUDE WARNING IN CASE PSF, SCIENCE AND REFS HAVE DIFFERENT SIZES
        #####################################################
            
    def PSFsub(self,
                pca_number,
                mask):
        
        self.pca_number = pca_number
        
        mask_in = round(mask / (self.data_hdr["CDELT1"]*3600))
        self.mask = create_mask((2*self.size+1,2*self.size+1), (mask_in, None))
        self.residuals = np.zeros_like(self.science)
        for k in range(len(self.science)):
            s = self.science[k]*self.mask
            r = self.refs_dict[:,k,:,:]*self.mask

            psf_model_k, residuals_k = apply_PCA(pca_number, s, r)
            self.residuals[k]=residuals_k

        hdul = fits.HDUList(fits.PrimaryHDU())
        hdul.append(fits.ImageHDU(self.residuals,header=self.data_hdr, name="RES"))
        hdul.writeto(self.output_dir + f'/Residuals/Residuals_{self.band}.fits', overwrite=True)
            
        #val = {"1A":0.0006, "1B":1e-3, "1C":4e-4, "2A":7e-4, "2B":7e-4, "2C":6e-4, "3A":3e-4, "3B":3e-4, "3C":250, "4A":150}
        fig, ax = plt.subplots(ncols=5, nrows=10, figsize=(10, 20))
        ax = np.array(ax).flatten()
        j=0

        for j in range(50):
            ax[j].imshow(self.residuals[8*j],
                             origin="lower", cmap="RdBu_r", vmin=-np.max(self.residuals[8*j]), vmax=np.max(self.residuals[8*j]))
            ax[j].text(0, 1, "$\lambda=$%.3f"%self.wvl[8*j], size=10, color="black")
            ax[j].set_xticks([])
            ax[j].set_yticks([])


        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.15, hspace=None)
        fig.savefig(self.output_dir + f'/Img/PCA_subtraction_wavelengths_{self.band}.pdf')
            
            
    def SNR(self,
                b_sep_lit,
                b_pa_lit):
        
        self.b_sep_lit = b_sep_lit
        self.ap_pix = interp1d(FWHM_wvl, FWHM_miri*0.75)(self.wvl) / self.pixelsize
        
        self.ap_pos = polar_to_cartesian(np.zeros((2*self.size+1,2*self.size+1)), sep = b_sep_lit/self.pixelsize, ang = MRS_PA[self.band[0]]) # Y, X
        shifts = np.loadtxt(self.output_dir + f'/Image_shifts/Shift_{self.band}.txt', skiprows=1)
        self.ap_pos += shifts
        print ('[WARNING]\t[The SNR aperture is hard coded on the location if GQ Lup B]')
        
        residuals_mean = np.nanmedian(self.residuals, axis=0)
        offset=None
        
        hdul = fits.HDUList(fits.PrimaryHDU())
        hdul.append(fits.ImageHDU(residuals_mean,header=self.data_hdr, name="RES MED"))
        hdul.writeto(self.output_dir + f'/Residuals/Median_residuals_{self.band}.fits', overwrite=True)
        
        _, _, snr, fpf = false_alarm(image=residuals_mean,
                            x_pos = self.ap_pos[1],
                            y_pos = self.ap_pos[0],
                            size = self.ap_pix[int(len(self.ap_pix)/2)],
                            ignore = True)

        print ('SNR = ', snr, ' for an aperture with radius ', self.ap_pix[int(len(self.ap_pix)/2)], ' pixels placed at ', self.ap_pos)
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(residuals_mean, origin="lower", cmap="RdBu_r", vmin=-np.max(residuals_mean), vmax=np.max(residuals_mean))
        ax.plot(self.ap_pos[1],self.ap_pos[0],'o',color='white', markersize=5)
        ax.text(14, 16, "SNR=%.1f"%snr, size=10, color="black")
        ax.text(0, 1, f"Band {self.band}", size=10, color="black")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.15, hspace=None)
        fig.savefig(self.output_dir + f'/Img/Median_residuals_{self.band}.pdf')
        
        
        snr_spectrum = []

        for k in range(len(self.residuals)):
            residuals_k = self.residuals[k]

            sum_ap, noise, snr, fpf = false_alarm(residuals_k, self.ap_pos[1],self.ap_pos[0],self.ap_pix[k],ignore=True)
            snr_spectrum.append(snr)

        snr_spectrum = np.array(snr_spectrum)
    
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(9, 4))
        ax.plot(self.wvl, snr_spectrum)
        ax.set_xlabel("$\lambda$ [$\mu m$]", size=18)
        ax.set_ylabel("SNR", size=18)
        ax.tick_params(labelsize=15)
        ax.set_ylim(0,np.max(snr_spectrum)+2)
        ax.set_xlim(self.wvl[0]-0.01, self.wvl[-1]+0.01)

        fig.subplots_adjust(left=0.08, bottom=0.16, right=0.99, top=0.99, wspace=0.15, hspace=None)
        fig.savefig(self.output_dir + f'/Img/SNR_{self.band}.pdf')
    


    def Contrast_spectrum(self):
        
        b_sep_from_center, b_PA_from_center = cartesian_to_polar(center = (self.size, self.size),
                                                                y_pos = self.ap_pos[0],
                                                                x_pos = self.ap_pos[1])
        
        fig, ax = plt.subplots(ncols=2, nrows=5, figsize=(5, 11))
        ax = np.array(ax).flatten()

        self.contrast_spectrum = []
        j=0
        for k in tqdm(range(len(self.science))):
            #####################################################
            ## TODO: What is the [:-10] at the end of the following line??
            selected = select_annulus(self.residuals, b_sep_from_center/self.pixelsize-2*self.ap_pix[k], b_sep_from_center/self.pixelsize+2*self.ap_pix[k])[:-10]
            var_noise = float(np.var(selected))
            #####################################################


            #####################################################
            ## TODO: The aperture location in the _objective function is rounded. This is wrong I think
            min_result = minimize(fun=_objective,
                    x0 = np.array([6.]),
                    args=((b_sep_from_center, b_PA_from_center), self.science[k].reshape(1, 2*self.size+1, 2*self.size+1),self.refs_dict[:,k,:,:],self.psf[k].reshape(1, 2*self.size+1, 2*self.size+1), 3*self.ap_pix[k],self.pca_number, var_noise, self.mask),
                    method='Nelder-Mead',
                    tol=None,
                    options={'xatol': 0.01, 'fatol': float('inf')})
            #####################################################
            
            self.contrast_spectrum.append(min_result.x[0])
            
            science_no_planet_k = fake_planet(images=self.science[k].reshape(1, 2*self.size+1, 2*self.size+1),
                                            psf=self.psf[k].reshape(1, 2*self.size+1, 2*self.size+1),
                                            parang=np.array([0]),
                                            position=(b_sep_from_center, b_PA_from_center),
                                            magnitude=min_result.x[0],
                                            psf_scaling=-1)

            _, res_no_planet_k = apply_PCA(self.pca_number, science_no_planet_k[0]*self.mask, self.refs_dict[:,k,:,:]*self.mask)

            n = int(len(self.science)/5)
            if k%n==0 and j<10:
                ax[j].imshow(self.residuals[k], origin='lower', cmap='RdBu_r', vmin=-np.max(self.residuals[k]), vmax=np.max(self.residuals[k]))
                ax[j].plot(self.ap_pos[1],self.ap_pos[0],'o',color='white', markersize=1)
                ax[j].plot(10,10,'o',color='k', markersize=1)
                ax[j].set_xticks([])
                ax[j].set_yticks([])
                ax[j].text(0, 2, '$\lambda$ = %.2f'%self.wvl[k], size=10, color="black")

                ax[j+1].imshow(res_no_planet_k, origin='lower', cmap='RdBu_r', vmin=-np.max(self.residuals[k]), vmax=np.max(self.residuals[k]))
                ax[j+1].plot(self.ap_pos[1],self.ap_pos[0],'o',color='k', markersize=1)
                ax[j+1].set_xticks([])
                ax[j+1].set_yticks([])
                #ax[j+1].text(0, 2, 'Fake planet', size=10, color="black")
                ax[j+1].text(0, 2, 'mag = %.1f'%min_result.x[0], size=10, color="black")
                j+=2

        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.15, hspace=None)
        fig.savefig(self.output_dir + f'Img/Planet_removed_{self.band}.pdf')
