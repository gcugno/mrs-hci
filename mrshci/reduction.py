import os
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from tqdm import tqdm
import pandas as pd
import astropy.units as u
import pickle
from photutils import CircularAperture, aperture_photometry
from spectres import spectres
import jwst


from util import prep_dict_cube, prep_wvl_cube, find_star, apply_PCA, crop_science, crop_refs, crop_psf, _objective, remove_outliers
from fixed_values import *

import sys
sys.path.append('/Users/gcugno/Tools/PynPoint')
from pynpoint.util.image import create_mask, polar_to_cartesian, select_annulus, cartesian_to_polar
from pynpoint.util.analysis import false_alarm, fake_planet


class MRS_HCI_PCA:
    def __init__(self,
                 output_dir,
                 science_path,
                 refs_path,
                 refs_names,
                 psf_name,
                 band = "1A",
                ):

        self.output_dir = output_dir
        self.band = band
        print (f'[INFO]\t\t[This is Band {self.band}]')
                
        
        if not os.path.isdir(output_dir+'/Image_shifts'):
            os.mkdir(output_dir+'/Image_shifts')
        if not os.path.isdir(output_dir+'/Img'):
            os.mkdir(output_dir+'/Img')
        if not os.path.isdir(output_dir+'/Manipulated_images'):
            os.mkdir(output_dir+'/Manipulated_images')
        if not os.path.isdir(output_dir+'/Residuals'):
            os.mkdir(output_dir+'/Residuals')
        if not os.path.isdir(output_dir+'/Extraction'):
            os.mkdir(output_dir+'/Extraction')
        if not os.path.isdir(output_dir+'/SNR'):
            os.mkdir(output_dir+'/SNR')
            
            
        self.data_import = prep_dict_cube(science_path, 'SCIENCE')[self.band]
        self.wvl, self.data_hdr = prep_wvl_cube(science_path)
        self.wvl = self.wvl[self.band]
        self.data_hdr = self.data_hdr[self.band]
        self.refs_names = refs_names
        
        self.pixelsize= self.data_hdr["CDELT1"]*3600
        print ('[INFO]\t\t[Pixelscale = ', self.pixelsize,']')
        
        
        self.refs_list = []
        for ref in refs_names:
            self.refs_list.append(prep_dict_cube(refs_path+ref+'/', ref)[self.band])
        
        self.psf_import = prep_dict_cube(refs_path+psf_name+'/', 'PSF')[self.band]
        
        return
    
    
    
    def prepare_cubes(self,
                     size = 10):
        
        self.size = size
        
        ## CROPPING the cubes
        self.science, D0 = crop_science(self.data_import, self.data_hdr, self.band, self.output_dir, size)
        self.refs_dict = crop_refs(self.refs_list, self.refs_names, self.data_hdr, self.band, D0, self.output_dir, size)
        self.psf = crop_psf(self.psf_import, self.data_hdr, self.band, self.output_dir, size)
        print ('[DONE]\t\t[All the data have been cropped]')

            
        # SAVE the new cubes, squared shapes
        hdul = fits.HDUList(fits.PrimaryHDU())
        hdul.append(fits.ImageHDU(self.refs_dict, name="REF"))
        hdul.writeto(self.output_dir + f'/Manipulated_images/References_{self.band}.fits', overwrite=True)

        hdul = fits.HDUList(fits.PrimaryHDU())
        hdul.append(fits.ImageHDU(self.science, name="SCI"))
        hdul.writeto(self.output_dir + f'/Manipulated_images/Science_{self.band}.fits', overwrite=True)

        hdul = fits.HDUList(fits.PrimaryHDU())
        hdul.append(fits.ImageHDU(self.psf, name="PSF"))
        hdul.writeto(self.output_dir + f'/Manipulated_images/PSF_{self.band}.fits', overwrite=True)
        print ('[DONE]\t\t[All the data saved in fits files]')
        
        # GIVE error if cubes are not of the same shape
        if np.shape(self.science) != np.shape(self.psf) or np.shape(self.science) != np.shape(self.refs_dict)[1:]:
            print ('[ERROR]\t\t[Shape of science = ', np.shape(self.science),']')
            print ('[ERROR]\t\t[Shape of PSF = ', np.shape(self.psf),']')
            print ('[ERROR]\t\t[Shape of references = ', np.shape(self.refs_dict),']')
            raise ValueError('[ERROR]\t\t[Arrays have different sizes]')
        
            
    def PSFsub(self,
                pca_number,
                mask):
        
        # Define number of pc within class
        self.pca_number = pca_number
        
        # Define the residuals cube as an array of zeros temporarily
        self.residuals = np.zeros_like(self.science)
        
        # Mask the central region of the image
        mask_in = round(mask / (self.data_hdr["CDELT1"]*3600))
        self.mask = create_mask((2*self.size+1,2*self.size+1), (mask_in, None))
        for k in range(len(self.science)):
            s = self.science[k]*self.mask
            r = self.refs_dict[:,k,:,:]*self.mask

            psf_model_k, residuals_k = apply_PCA(pca_number, s, r)
            self.residuals[k]=residuals_k

        # Save residuals in fits file
        hdul = fits.HDUList(fits.PrimaryHDU())
        hdul.append(fits.ImageHDU(self.residuals,header=self.data_hdr, name="RES"))
        hdul.writeto(self.output_dir + f'/Residuals/Residuals_{self.band}.fits', overwrite=True)
            
        # Plot the residuals every 8 channels
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
                b_pa_lit,
                r_in_FWHM=0.5):
        
        self.b_sep = b_sep_lit
        self.ap_pix = (0.033 * self.wvl + 0.106)/ self.pixelsize * r_in_FWHM
        
        
        self.ap_pos = polar_to_cartesian(np.zeros((2*self.size+1,2*self.size+1)), sep = b_sep_lit/self.pixelsize, ang = MRS_PA[self.band[0]]) # Y, X
        shifts = np.loadtxt(self.output_dir + f'/Image_shifts/Shift_{self.band}.txt', skiprows=1)
        self.ap_pos += shifts
        print ('[WARNING]\t[The SNR aperture is hard coded on the location if GQ Lup B]')
        print ('[INFO]\t\t[Aperture position = ', self.ap_pos,']')
        
        residuals_mean = np.nanmedian(self.residuals, axis=0)
        self.vval = np.max(residuals_mean)
        offset=None
        
        hdul = fits.HDUList(fits.PrimaryHDU())
        hdul.append(fits.ImageHDU(residuals_mean,header=self.data_hdr, name="RES MED"))
        hdul.writeto(self.output_dir + f'/Residuals/Median_residuals_{self.band}.fits', overwrite=True)
        
        try:
            _, _, snr, fpf = false_alarm(image=residuals_mean,
                            x_pos = self.ap_pos[1],
                            y_pos = self.ap_pos[0],
                            size = self.ap_pix[int(len(self.ap_pix)/2)],
                            ignore = True)

            print ('[RESULT]\t[SNR = %.1f in an aperture of radius %.1f pixels]'%(snr, np.mean(self.ap_pix)))
            pd_df2 = pd.DataFrame(np.array([snr]))
            pd_df2.to_csv(self.output_dir + f'/SNR/SNR_{self.band}.txt', index=False, header=('SNR'), sep='\t')
        
        except:
            pd_df2 = pd.DataFrame(np.array(['0']))
            pd_df2.to_csv(self.output_dir + f'/SNR/SNR_{self.band}.txt', index=False, header=('SNR'), sep='\t')
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(residuals_mean, origin="lower", cmap="RdBu_r", vmin=-self.vval, vmax=self.vval)
        ax.plot(self.ap_pos[1],self.ap_pos[0],'o',color='white', markersize=5)
        try:
            ax.text(14, 16, "SNR=%.1f"%snr, size=10, color="black")
        except:
            pass
        ax.text(0, 1, f"Band {self.band}", size=10, color="black")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.15, hspace=None)
        fig.savefig(self.output_dir + f'/Img/Median_residuals_{self.band}.pdf')
        
        try:
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
        except:
            pass
           
           
           
           
    def SNR_fit(self,
                b_sep_lit,
                b_pa_lit,
                r_in_FWHM=0.5):
        
        # Define aperture radius based on Law+2023
        self.ap_pix = (0.033 * self.wvl + 0.106)/ self.pixelsize * r_in_FWHM
        
        # Take the median of the residuals cube
        residuals_mean = np.nanmedian(self.residuals, axis=0)
        # Find the maximum of the median for plotting purposes
        self.vval = np.max(residuals_mean)
        #offset=None
        
        # SAVE the median of the residuals
        hdul = fits.HDUList(fits.PrimaryHDU())
        hdul.append(fits.ImageHDU(residuals_mean,header=self.data_hdr, name="RES MED"))
        hdul.writeto(self.output_dir + f'/Residuals/Median_residuals_{self.band}.fits', overwrite=True)
        
        # Find the companion position by fitting a 2D gaussian to the residuals
        self.ap_pos = find_star(residuals_mean)
        print ('[INFO]\t\t[Aperture from fit in band ', self.band, ' = (', int(self.ap_pos[0]*100)/100, ', ', int(self.ap_pos[1]*100)/100 ,')]')
        
        # Calculate and save in txt file the SNR of the detection
        try:
            _, _, snr, fpf = false_alarm(image=residuals_mean,
                            x_pos = self.ap_pos[1],
                            y_pos = self.ap_pos[0],
                            size = self.ap_pix[int(len(self.ap_pix)/2)],
                            ignore = True)

            print ('[RESULT]\t[SNR = %.1f in an aperture of radius %.1f pixels]'%(snr, np.mean(self.ap_pix)))
            pd_df2 = pd.DataFrame(np.array([snr]))
            pd_df2.to_csv(self.output_dir + f'/SNR/SNR_{self.band}.txt', index=False, header=('SNR'), sep='\t')
        
        # If the SNR can not be calculated, save a 0 in the txt file
        except:
            pd_df2 = pd.DataFrame(np.array(['0']))
            pd_df2.to_csv(self.output_dir + f'/SNR/SNR_{self.band}.txt', index=False, header=('SNR'), sep='\t')
        
        
        # Plot the residuals including writing the calculated SNR
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(residuals_mean, origin="lower", cmap="RdBu_r", vmin=-self.vval, vmax=self.vval)
        ax.plot(self.ap_pos[1],self.ap_pos[0],'*',color='white', markersize=5)
        try:
            ax.text(14, 16, "SNR=%.1f"%snr, size=10, color="black")
        except:
            pass
        ax.text(0, 1, f"Band {self.band}", size=10, color="black")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.15, hspace=None)
        fig.savefig(self.output_dir + f'/Img/Median_residuals_{self.band}.pdf')
        
        # Try to estimate SNR in every single channel. If it works, good, otherwise, pass
        try:
            snr_spectrum = []
            
            # Iterate over every single channel
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
        except:
            pass
        
        
        

    def Contrast_spectrum(self):
        
        # Find the polar coordinates of the companion.
        self.b_sep_from_center, b_PA_from_center = cartesian_to_polar(center = (self.size, self.size),
                                                                y_pos = self.ap_pos[0],
                                                                x_pos = self.ap_pos[1])
        
        # Initiate the plot of the residuals with and without the companion
        fig, ax = plt.subplots(ncols=2, nrows=5, figsize=(5, 11))
        ax = np.array(ax).flatten()

        self.contrast_spectrum = []
        self.science_no_planet = []
        j=0
        for k in tqdm(range(len(self.science))):
            selected = select_annulus(self.residuals[k], self.b_sep_from_center-3*self.ap_pix[k], self.b_sep_from_center+3*self.ap_pix[k], (self.ap_pos[0], self.ap_pos[1]), 3*self.ap_pix[k])
            var_noise = float(np.var(selected))


            min_result = minimize(fun=_objective,
                    x0 = np.array([6.]),
                    args=((self.b_sep_from_center, b_PA_from_center), self.science[k].reshape(1, 2*self.size+1, 2*self.size+1),self.refs_dict[:,k,:,:],self.psf[k].reshape(1, 2*self.size+1, 2*self.size+1), 3*self.ap_pix[k],self.pca_number, var_noise, self.mask),
                    method='Nelder-Mead',
                    tol=None,
                    options={'xatol': 0.01, 'fatol': float('inf')})
            
            self.contrast_spectrum.append(min_result.x[0])
            
            science_no_planet_k = fake_planet(images=self.science[k].reshape(1, 2*self.size+1, 2*self.size+1),
                                            psf=self.psf[k].reshape(1, 2*self.size+1, 2*self.size+1),
                                            parang=np.array([0]),
                                            position=(self.b_sep_from_center, b_PA_from_center),
                                            magnitude=min_result.x[0],
                                            psf_scaling=-1)
                                            
            self.science_no_planet.append(science_no_planet_k)

            _, res_no_planet_k = apply_PCA(self.pca_number, science_no_planet_k[0]*self.mask, self.refs_dict[:,k,:,:]*self.mask)

            n = int(len(self.science)/5)
            if k%n==0 and j<10:
                ax[j].imshow(self.residuals[k], origin='lower', cmap='RdBu_r', vmin=-self.vval, vmax=self.vval)
                ax[j].plot(self.ap_pos[1],self.ap_pos[0],'o',color='white', markersize=1)
                ax[j].plot(10,10,'o',color='k', markersize=1)
                ax[j].set_xticks([])
                ax[j].set_yticks([])
                ax[j].text(0, 2, '$\lambda$ = %.2f'%self.wvl[k], size=10, color="black")

                ax[j+1].imshow(res_no_planet_k, origin='lower', cmap='RdBu_r', vmin=-self.vval, vmax=self.vval)
                ax[j+1].plot(self.ap_pos[1],self.ap_pos[0],'o',color='k', markersize=1)
                ax[j+1].set_xticks([])
                ax[j+1].set_yticks([])
                ax[j+1].text(0, 2, 'mag = %.1f'%min_result.x[0], size=10, color="black")
                j+=2

        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.15, hspace=None)
        fig.savefig(self.output_dir + f'Img/Planet_removed_{self.band}.pdf')
        
        self.science_no_planet = np.array(self.science_no_planet)
        
        print ('[DONE]\t\t[Contrast spectrum was calculated for an aperture of size %.1f pix and exported]'%(3*np.mean(self.ap_pix)))




    def Estimate_uncertainties(self,
                                num_angles = 10):
            
        PAs = np.linspace(0,359,num_angles)
        
        self.offset_mean = []
        self.offset_std = []
        self.offset_all = []
        
        #j=0
        #fig, ax = plt.subplots(ncols=3, nrows=8, figsize=(5, 15))
        #ax = np.array(ax).flatten()
            
        for k in tqdm(range(len(self.science))):
            _, res_no_planet_k = apply_PCA(self.pca_number, self.science_no_planet[k][0]*self.mask, self.refs_dict[:,k,:,:]*self.mask)
            selected = select_annulus(res_no_planet_k, self.b_sep_from_center-3*self.ap_pix[k], self.b_sep_from_center+3*self.ap_pix[k], (self.ap_pos[0], self.ap_pos[1]), 3*self.ap_pix[k])
            #selected = select_annulus(res_no_planet_k, self.b_sep_from_center-3*self.ap_pix[k], self.b_sep_from_center+3*self.ap_pix[k])[:-10]
            var_noise = float(np.var(selected))

        
            offsets_k = []
            aperture_diff_k = []
            for pa_i, pa in enumerate(PAs):
                #print ('PA = ', pa)
                fake_pa = fake_planet(images = self.science_no_planet[k].reshape(1, 2*self.size+1, 2*self.size+1),
                                        psf = self.psf[k].reshape(1, 2*self.size+1, 2*self.size+1),
                                        parang = np.array([0]),
                                        position = (self.b_sep_from_center, pa),
                                        magnitude = self.contrast_spectrum[k],
                                        psf_scaling = 1)
                #print ('fake_pa shape = ', np.shape(fake_pa))
                                        
                min_result_pa = minimize(fun=_objective,
                                    x0 = np.array([self.contrast_spectrum[k]]),
                                    args=((self.b_sep_from_center, pa), fake_pa, self.refs_dict[:,k,:,:], self.psf[k].reshape(1, 2*self.size+1, 2*self.size+1), 3*self.ap_pix[k], self.pca_number, var_noise, self.mask, True),
                                    method='Nelder-Mead',
                                    tol=1e-3,
                                    options={'xatol': 0.01, 'fatol': float('inf')})
                                    
                contrast_retrieved = min_result_pa.x[0]
                #print ('contrasts = ', contrast_retrieved, '\t\t', self.contrast_spectrum[k])
                offsets_k.append(contrast_retrieved-self.contrast_spectrum[k])
                
                #_, res_empty = apply_PCA(self.pca_number, self.science_no_planet[k][0]*self.mask, self.refs_dict[:,k,:,:]*self.mask)
                #_, res_fake_pa = apply_PCA(self.pca_number, fake_pa[0]*self.mask, self.refs_dict[:,k,:,:]*self.mask)
                #print ('Res = ', res_fake_pa)
                
                #if pa_i%(num_angles/2)==0 and k==0:
                #if k==0:
                    #ap_pos_plot = polar_to_cartesian(res_fake_pa, sep = b_sep_from_center/pixelsize, ang = pa)
            
                    #ax[j].imshow(res_no_planet_k , origin='lower', cmap='RdBu_r', vmin=-self.vval, vmax=self.vval)
                    #ax[j].plot(ap_pos_plot[1],ap_pos_plot[0],'o',color='white', markersize=1)
                    #ax[j].set_xticks([])
                    #ax[j].set_yticks([])
                    #ax[j].text(0, 2, '$\lambda$ = %.2f'%self.wvl[k], size=10, color="black")
           
                    #ax[j+1].imshow(res_fake_pa, origin='lower', cmap='RdBu_r', vmin=-self.vval, vmax=self.vval)
                    #ax[j+1].set_xticks([])
                    #ax[j+1].set_yticks([])
                    #ax[j+1].text(0, 2, 'mag = %.1f'%contrast_retrieved, size=10, color="black")
                    #j+=3
            
            self.offset_mean.append(np.mean(offsets_k))
            self.offset_std.append(np.std(offsets_k))
            #fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.15, hspace=None)
            #fig.savefig(self.output_dir + f'/Img/Offsets_{self.band}.pdf')
            self.offset_all.append(offsets_k)
            
        self.offset_all = np.ravel(self.offset_all)
        
        fig, ax = plt.subplots(figsize=(10, 12))
        
        ax.hist(self.offset_all, bins=50)
        ax.set_xlim(-1.,1.)
        fig.savefig(self.output_dir + f'/Img/Histogram_offsets_{self.band}.pdf')
        
            

        #####################################################
        ## TODO: Move this after the calculation of the error took place as well
        pt2 = np.vstack((self.wvl, self.contrast_spectrum, self.offset_mean, self.offset_std))
        pd_df2 = pd.DataFrame(pt2.T)
        pd_df2.to_csv(self.output_dir + f'/Extraction/Contrast_{self.band}.txt', index=False, header=('wvl [um]', 'Contrast [mag]', 'Bias [mag]', 'Sys error [mag]'), sep='\t')
        #####################################################





    def Extract_spectrum(self,
                        spectrum_path,
                        outlier_thres = 5,
                        bin = 50):

        
        self.offset_mean = np.array(self.offset_mean)
        self.offset_std = np.array(self.offset_std)
        self.contrast_spectrum = np.array(self.contrast_spectrum)
        
        psf_flux = pickle.load(open(spectrum_path, 'rb'))[self.band]
        psf_err = psf_flux/200
        #if self.band == "2C":
        #    psf_flux = psf_flux[:1293]
        flux = psf_flux * 10**(-(self.contrast_spectrum-self.offset_mean)/2.5)*1000
        
        flux_err_psf = psf_err * 10**(-(self.contrast_spectrum-self.offset_mean)/2.5)*1000
        flux_err_up = psf_flux * 10**(-(self.contrast_spectrum - self.offset_mean - self.offset_std)/2.5)*1000-flux
        flux_err_up = np.sqrt(flux_err_up**2+flux_err_psf**2)
        
        flux_err_down = -psf_flux * 10**(-(self.contrast_spectrum - self.offset_mean + self.offset_std)/2.5)*1000+flux
        flux_err_down = np.sqrt(flux_err_down**2 + flux_err_psf**2)
        
        flux_err = np.maximum(flux_err_up, flux_err_down)
        
        # Export the sepctra
        pt2 = np.vstack((self.wvl, flux, flux_err))
        pd_df2 = pd.DataFrame(pt2.T)
        pd_df2.to_csv(self.output_dir + f'/Extraction/Flux_{self.band}.txt', index=False, header=('wvl [um]', 'Flux [mJy]', 'Flux err [mJy]'), sep='\t')
        
        
        # Call function that removes outliers
        wvl_clean, flux_clean, flux_err_clean = remove_outliers(self.wvl, flux, flux_err , outlier_thres)
        
        
        # Add residuals RMS after continuum subtraction to the error budget
        coeff = np.polyfit(wvl_clean, flux_clean, 1)
        cont = np.polyval(coeff, wvl_clean)
        
        rms = np.sqrt(np.mean((flux_clean - cont)**2))
        flux_err_clean = np.sqrt(flux_err_clean**2 + (flux_clean - cont)**2)
        
        
        # Export the sepctra without outliers
        pt2 = np.vstack((wvl_clean, flux_clean, flux_err_clean))
        pd_df2 = pd.DataFrame(pt2.T)
        pd_df2.to_csv(self.output_dir + f'/Extraction/Flux_clean_{self.band}.txt', index=False, header=('wvl [um]', 'Flux [mJy]', 'Flux err [mJy]'), sep='\t')
        print ('[DONE]\t\t[Planet spectrum was calculated and exported]')
        
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(9, 4))
        ax.plot(self.wvl, flux, color='dodgerblue', alpha=0.8, label=f'Flux in {self.band}')
        ax.fill_between(self.wvl, flux-flux_err, flux+flux_err, color='lightblue', alpha=0.3)
        ax.plot(wvl_clean, flux_clean, color='k', lw=0.5)
        ax.fill_between(wvl_clean, flux_clean-flux_err_clean, flux_clean+flux_err_clean, color='silver', alpha=0.3)

        ax.set_xlabel("$\lambda$ [$\mu m$]", size=18)
        ax.set_ylabel("Flux [mJy]", size=18)
        ax.tick_params(labelsize=15)
        ax.set_ylim(np.median(flux_clean)-10*np.std(flux_clean),np.median(flux_clean)+10*np.std(flux_clean))
        ax.set_xlim(self.wvl[0]-0.01, self.wvl[-1]+0.01)
        ax.legend(prop={'size':15})

        fig.subplots_adjust(left=0.11, bottom=0.16, right=0.99, top=0.97, wspace=0.15, hspace=None)
        fig.savefig(self.output_dir + f'/Img/Flux_{self.band}.pdf')



        flux_cgs = (flux_clean*u.mJy).to(u.W/u.m**2/u.micron, equivalencies=u.spectral_density(wvl_clean*u.micron)).value
        flux_err_cgs = (flux_err_clean*u.mJy).to(u.W/u.m**2/u.micron, equivalencies=u.spectral_density(wvl_clean*u.micron)).value
        
        pt2 = np.vstack((wvl_clean, flux_cgs, flux_err_cgs))
        pd_df2 = pd.DataFrame(pt2.T)
        pd_df2.to_csv(self.output_dir + f'/Extraction/Flux_cgs_{self.band}.txt', index=False, header=('wvl [um]', 'Flux [W m^-2 um^-1]', 'Flux err [W m^-2 um^-1]'), sep='\t')
        
        
        if bin%2!=0:
            raise ValueError('Chose an even bin number')
        
        wvl_bin = wvl_clean[int(bin/2)::bin][:-1]
        
        flux_cgs_bin, flux_err_cgs_bin = spectres(wvl_bin, wvl_clean, flux_cgs, flux_err_cgs)
        
        pt2 = np.vstack((wvl_bin, flux_cgs_bin, flux_err_cgs_bin))
        pd_df2 = pd.DataFrame(pt2.T)
        pd_df2.to_csv(self.output_dir + f'/Extraction/Flux_cgs_bin{str(bin)}_{self.band}.txt', index=False, header=('wvl [um]', 'Flux [W m^-2 um^-1]', 'Flux err [W m^-2 um^-1]'), sep='\t')
        print ('\n\n')









class MRS_HCI_simplesub:
    def __init__(self,
                 output_dir,
                 science_path,
                 refs_path,
                 refs_names,
                 psf_name,
                 band = "3A",
                ):

        self.output_dir = output_dir
        self.band = band
                
        
        if not os.path.isdir(output_dir+'/Image_shifts'):
            os.mkdir(output_dir+'/Image_shifts')
        if not os.path.isdir(output_dir+'/Img'):
            os.mkdir(output_dir+'/Img')
        if not os.path.isdir(output_dir+'/Manipulated_images'):
            os.mkdir(output_dir+'/Manipulated_images')
        if not os.path.isdir(output_dir+'/Residuals'):
            os.mkdir(output_dir+'/Residuals')
        if not os.path.isdir(output_dir+'/Extraction'):
            os.mkdir(output_dir+'/Extraction')
            
            
        self.data_import = prep_dict_cube(science_path, 'SCIENCE')[self.band]
        self.wvl, self.data_hdr = prep_wvl_cube(science_path)
        self.wvl = self.wvl[self.band]
        self.data_hdr = self.data_hdr[self.band]
        self.refs_names = refs_names
        
        self.pixelsize= self.data_hdr["CDELT1"]*3600
        print ('[INFO]\t\t[Pixelscale = ', self.pixelsize,']')
        
        
        self.refs_list = []
        for ref in refs_names:
            self.refs_list.append(prep_dict_cube(refs_path+ref+'/', ref)[self.band])
        
        self.psf_import = prep_dict_cube(refs_path+psf_name+'/', 'PSF')[self.band]
        
        #####################################################
        # WHAT IS GOING ON WITH 2C AND THE DIMENSION?
        print ('[WARNING]\t[Verify dimensions in channel 2C]')
        #####################################################
        
        
        return



    def prepare_cubes(self,
                     size = 10):
        
        self.size = size
        
        ## CROPPING the cubes
        self.science, D0 = crop_science(self.data_import, self.data_hdr, self.band, self.output_dir, size)
        self.refs_dict = crop_refs(self.refs_list, self.refs_names, self.data_hdr, self.band, D0, self.output_dir, size)[0]
        self.psf = crop_psf(self.psf_import, self.data_hdr, self.band, self.output_dir, size)
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
        print ('[DONE]\t\t[All the data saved in fits files]')
        
        if np.shape(self.science) != np.shape(self.psf) or np.shape(self.science) != np.shape(self.refs_dict):
            print ('[ERROR]\t\t[Shape of science = ', np.shape(self.science),']')
            print ('[ERROR]\t\t[Shape of PSF = ', np.shape(self.psf),']')
            print ('[ERROR]\t\t[Shape of references = ', np.shape(self.refs_dict),']')
            raise ValueError('[ERROR]\t\t[Arrays have different sizes]')
            
    
    
    def PSFsub(self,
                mask):
        
        
        mask_in = round(mask / (self.data_hdr["CDELT1"]*3600))
        self.mask = create_mask((2*self.size+1,2*self.size+1), (mask_in, None))
        self.residuals = np.zeros_like(self.science)
        #####################################################
        # WHAT IS GOING ON WITH 2C AND THE DIMENSION?
        print ('[WARNING]\t[Aperture is currently fixed! Make it variable!]')
        #####################################################
        shifts = np.loadtxt(self.output_dir + f'/Image_shifts/Shift_{self.band}.txt', skiprows=1)
        for k in range(len(self.science)):
            ap = CircularAperture((self.size+shifts[1], self.size+shifts[0]), r=6.)
            fr = aperture_photometry(self.refs_dict[k], ap)
            fd = aperture_photometry(self.science[k], ap)
        
            norm = fd['aperture_sum']/fr['aperture_sum']
            self.refs_dict[k] = norm * self.refs_dict[k]
            
            s = self.science[k]*self.mask
            r = self.refs_dict[k]*self.mask

            self.residuals[k] = s - r

        hdul = fits.HDUList(fits.PrimaryHDU())
        hdul.append(fits.ImageHDU(self.residuals,header=self.data_hdr, name="RES"))
        hdul.writeto(self.output_dir + f'/Residuals/Residuals_{self.band}.fits', overwrite=True)
            
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
        fig.savefig(self.output_dir + f'/Img/Simplesub_subtraction_wavelengths_{self.band}.pdf')



    def SNR(self,
                b_sep_lit,
                b_pa_lit,
                r_in_FWHM=0.5):
        
        self.b_sep = b_sep_lit
        self.ap_pix = (0.033 * self.wvl + 0.106)/ self.pixelsize * r_in_FWHM
        
        
        self.ap_pos = polar_to_cartesian(np.zeros((2*self.size+1,2*self.size+1)), sep = b_sep/self.pixelsize, ang = MRS_PA[self.band[0]]) # Y, X
        shifts = np.loadtxt(self.output_dir + f'/Image_shifts/Shift_{self.band}.txt', skiprows=1)
        self.ap_pos += shifts
        print ('[WARNING]\t[The SNR aperture is hard coded on the location if GQ Lup B]')
        print ('[INFO]\t\t[Aperture position = ', self.ap_pos,']')
        print ('[INFO]\t\t[Aperture size = ', np.mean(self.ap_pix),']')
        
        residuals_mean = np.nanmedian(self.residuals, axis=0)
        self.vval = np.max(residuals_mean)
        offset=None
        
        hdul = fits.HDUList(fits.PrimaryHDU())
        hdul.append(fits.ImageHDU(residuals_mean,header=self.data_hdr, name="RES MED"))
        hdul.writeto(self.output_dir + f'/Residuals/Median_residuals_{self.band}.fits', overwrite=True)
        
        try:
            _, _, snr, fpf = false_alarm(image=residuals_mean,
                            x_pos = self.ap_pos[1],
                            y_pos = self.ap_pos[0],
                            size = self.ap_pix[int(len(self.ap_pix)/2)],
                            ignore = True)

            print ('[RESULT]\t[SNR = %.1f in an aperture of radius %.1f pixels]'%(snr, np.mean(self.ap_pix)))
            pd_df2 = pd.DataFrame(np.array([snr]))
            pd_df2.to_csv(self.output_dir + f'/SNR/SNR_{self.band}.txt', index=False, header=('SNR'), sep='\t')
            
        except:
            pd_df2 = pd.DataFrame(np.array(['0']))
            pd_df2.to_csv(self.output_dir + f'/SNR/SNR_{self.band}.txt', index=False, header=('SNR'), sep='\t')
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(residuals_mean, origin="lower", cmap="RdBu_r", vmin=-self.vval, vmax=self.vval)
        ax.plot(self.ap_pos[1],self.ap_pos[0],'o',color='white', markersize=5)
        #ax.text(14, 16, "SNR=%.1f"%snr, size=10, color="black")
        ax.text(0, 1, f"Band {self.band}", size=10, color="black")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.15, hspace=None)
        fig.savefig(self.output_dir + f'/Img/Median_residuals_{self.band}.pdf')
        
        
        '''snr_spectrum = []

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
        fig.savefig(self.output_dir + f'/Img/SNR_{self.band}.pdf')'''



    def Contrast_spectrum(self):
        
        self.b_sep_from_center, b_PA_from_center = cartesian_to_polar(center = (self.size, self.size),
                                                                y_pos = self.ap_pos[0],
                                                                x_pos = self.ap_pos[1])
        
        fig, ax = plt.subplots(ncols=2, nrows=5, figsize=(5, 11))
        ax = np.array(ax).flatten()

        self.contrast_spectrum = []
        self.science_no_planet = []
        j=0
        for k in tqdm(range(len(self.science))):
            selected = select_annulus(self.residuals[k], self.b_sep_from_center-1*self.ap_pix[k], self.b_sep_from_center+1*self.ap_pix[k], (self.ap_pos[0], self.ap_pos[1]), 2*self.ap_pix[k])
            var_noise = float(np.var(selected))


            min_result = minimize(fun=_objective,
                    x0 = np.array([3.]),
                    args=((self.b_sep_from_center, b_PA_from_center), self.science[k].reshape(1, 2*self.size+1, 2*self.size+1),self.refs_dict[k],self.psf[k].reshape(1, 2*self.size+1, 2*self.size+1), 0.5*self.ap_pix[k],None, var_noise, self.mask),
                    method='Nelder-Mead',
                    tol=None,
                    options={'xatol': 0.01, 'fatol': float('inf')})
            
            self.contrast_spectrum.append(min_result.x[0])
            
            science_no_planet_k = fake_planet(images=self.science[k].reshape(1, 2*self.size+1, 2*self.size+1),
                                            psf=self.psf[k].reshape(1, 2*self.size+1, 2*self.size+1),
                                            parang=np.array([0]),
                                            position=(self.b_sep_from_center, b_PA_from_center),
                                            magnitude=min_result.x[0],
                                            psf_scaling=-1)
                                            
            self.science_no_planet.append(science_no_planet_k[0])

            res_no_planet_k = self.science_no_planet[k]*self.mask - self.refs_dict[k]*self.mask

            n = int(len(self.science)/5)
            if k%n==0 and j<10:
                ax[j].imshow(self.residuals[k], origin='lower', cmap='RdBu_r', vmin=-self.vval, vmax=self.vval)
                ax[j].plot(self.ap_pos[1],self.ap_pos[0],'o',color='white', markersize=1)
                ax[j].plot(10,10,'o',color='k', markersize=1)
                ax[j].set_xticks([])
                ax[j].set_yticks([])
                ax[j].text(0, 2, '$\lambda$ = %.2f'%self.wvl[k], size=10, color="black")

                ax[j+1].imshow(res_no_planet_k, origin='lower', cmap='RdBu_r', vmin=-self.vval, vmax=self.vval)
                ax[j+1].plot(self.ap_pos[1],self.ap_pos[0],'o',color='k', markersize=1)
                ax[j+1].set_xticks([])
                ax[j+1].set_yticks([])
                ax[j+1].text(0, 2, 'mag = %.1f'%min_result.x[0], size=10, color="black")
                j+=2

        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.15, hspace=None)
        fig.savefig(self.output_dir + f'Img/Planet_removed_{self.band}.pdf')
        
        self.science_no_planet = np.array(self.science_no_planet)
        
        print ('[DONE]\t\t[Contrast spectrum was calculated for an aperture of size %.1f pix and exported]'%(3*np.mean(self.ap_pix)))

        #####################################################
        ## TODO: Move this after the calculation of the error took place as well
        self.offset_mean = np.zeros_like(self.contrast_spectrum)
        self.offset_std = np.zeros_like(self.contrast_spectrum)
        pt2 = np.vstack((self.wvl, self.contrast_spectrum, self.offset_mean, self.offset_std))
        pd_df2 = pd.DataFrame(pt2.T)
        pd_df2.to_csv(self.output_dir + f'/Extraction/Contrast_{self.band}.txt', index=False, header=('wvl [um]', 'Contrast [mag]', 'Bias [mag]', 'Sys error [mag]'), sep='\t')
        #####################################################







    def Extract_spectrum(self,
                        spectrum_path,
                        outlier_thres = 5):

        
        self.offset_mean = np.array(self.offset_mean)
        self.offset_std = np.array(self.offset_std)
        self.contrast_spectrum = np.array(self.contrast_spectrum)
        
        psf_flux = pickle.load(open(spectrum_path, 'rb'))[self.band]
        if self.band == "2C":
            psf_flux = psf_flux[:1293]
        flux = psf_flux * 10**(-(self.contrast_spectrum-self.offset_mean)/2.5)*1000
        flux_err_up = psf_flux * 10**(-(self.contrast_spectrum - self.offset_mean - self.offset_std)/2.5)*1000-flux
        flux_err_down = -psf_flux * 10**(-(self.contrast_spectrum - self.offset_mean + self.offset_std)/2.5)*1000+flux
        flux_err = np.minimum(flux_err_up, flux_err_down)
        
        # Export the sepctra
        pt2 = np.vstack((self.wvl, flux, flux_err))
        pd_df2 = pd.DataFrame(pt2.T)
        pd_df2.to_csv(self.output_dir + f'/Extraction/Flux_{self.band}.txt', index=False, header=('wvl [um]', 'Flux [mJy]', 'Flux err [mJy]'), sep='\t')
        
        # Call function that removes residual fringe
        flux = jwst.residual_fringe.utils.fit_residual_fringes_1d(flux, self.wvl, channel=self.band[0],
                                                          dichroic_only=False, max_amp=None)
                                                          
        # Call function that removes outliers
        wvl_clean, flux_clean, flux_err_clean = remove_outliers(self.wvl, flux, flux_err , outlier_thres)
        
        
        # Export the sepctra without outliers
        pt2 = np.vstack((wvl_clean, flux_clean, flux_err_clean))
        pd_df2 = pd.DataFrame(pt2.T)
        pd_df2.to_csv(self.output_dir + f'/Extraction/Flux_clean_{self.band}.txt', index=False, header=('wvl [um]', 'Flux [mJy]', 'Flux err [mJy]'), sep='\t')
        print ('[DONE]\t\t[Planet spectrum was calculated and exported]')
        
        
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(9, 4))
        ax.plot(self.wvl, flux, color='dodgerblue', alpha=0.8, label=f'Flux in {self.band}')
        ax.fill_between(self.wvl, flux-flux_err, flux+flux_err, color='lightblue', alpha=0.3)
        ax.plot(wvl_clean, flux_clean, color='k', lw=0.5)
        ax.fill_between(wvl_clean, flux_clean-flux_err_clean, flux_clean+flux_err_clean, color='silver', alpha=0.3)

        ax.set_xlabel("$\lambda$ [$\mu m$]", size=18)
        ax.set_ylabel("Flux [mJy]", size=18)
        ax.tick_params(labelsize=15)
        ax.set_ylim(np.median(flux_clean)-10*np.std(flux_clean),np.median(flux_clean)+10*np.std(flux_clean))
        ax.set_xlim(self.wvl[0]-0.01, self.wvl[-1]+0.01)
        ax.legend(prop={'size':15})

        fig.subplots_adjust(left=0.11, bottom=0.16, right=0.99, top=0.97, wspace=0.15, hspace=None)
        fig.savefig(self.output_dir + f'/Img/Flux_{self.band}.pdf')



        flux_cgs = (flux_clean*u.mJy).to(u.W/u.m**2/u.micron, equivalencies=u.spectral_density(wvl_clean*u.micron)).value
        flux_err_cgs = (flux_err_clean*u.mJy).to(u.W/u.m**2/u.micron, equivalencies=u.spectral_density(wvl_clean*u.micron)).value
        
        pt2 = np.vstack((wvl_clean, flux_cgs, flux_err_cgs))
        pd_df2 = pd.DataFrame(pt2.T)
        pd_df2.to_csv(self.output_dir + f'/Extraction/Flux_cgs_{self.band}.txt', index=False, header=('wvl [um]', 'Flux [W m^-2 um^-1]', 'Flux err [W m^-2 um^-1]'), sep='\t')
