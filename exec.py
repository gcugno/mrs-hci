from reduction import MRS_HCI_PCA, MRS_HCI_simplesub
import numpy as np


thres_out = {"1A":5, "1B":5, "1C":5,"2A":5, "2B":5, "2C":8, "3A":8, "3B":8, "3C":8}
size_bands = {"1A":9, "1B":9, "1C":9, "2A":8, "2B":8, "2C":8, "3A":8, "3B":8, "3C":8}


bands = np.array(["1A","1B","1C","2A","2B","2C"])
bands = np.array(["2C"])


for band in bands:
    gqlup_red = MRS_HCI_PCA(output_dir = '/Users/gcugno/Science/JWST/MRS/GQLup/Results_new/',
              science_path = "/Users/gcugno/Science/JWST/MRS/GQLup/Raw_data/GQLup/",
              refs_path = "/Users/gcugno/Science/JWST/MRS/GQLup/Raw_data/",
              refs_names = ['1050_3', '1050_5', '1050_9', '1524_1', '1524_2', '1524_3', '1524_4', '1524_6', '1524_7', '1524_8', '1524_17', '1536_22', '1536_23', '1538_1', '1538_2', '1538_3','1640_10'], # ALL
              #refs_names = ['1050_3', '1050_5', '1524_1', '1524_2', '1524_3', '1524_4', '1524_6', '1524_7', '1524_8', '1524_17', '1538_1', '1640_10'],
              #refs_names = ['1050_3', '1050_9', '1524_17', '1536_22', '1536_23', '1538_1', '1538_2', '1538_3', '1640_10'], # OLD VERSION
              psf_name = '1536_22',#main:1536_22    other_1538_1
              band = band)
              


    gqlup_red.prepare_cubes(size=size_bands[band])

    gqlup_red.PSFsub(pca_number = 10,
           mask = 0.5)

    gqlup_red.SNR_fit(b_sep_lit = 0.708,
                    b_pa_lit = 279,
                    r_in_FWHM=0.5)

    gqlup_red.Contrast_spectrum()

    gqlup_red.Estimate_uncertainties(num_angles=10)

    gqlup_red.Extract_spectrum(spectrum_path = "/Users/gcugno/Science/JWST/MRS/GQLup/Raw_data/spectrum_1536_obs22",
                                outlier_thres = thres_out[band])



bands = np.array([])
for band in bands:
    
    gqlup_red = MRS_HCI_simplesub(output_dir = '/Users/gcugno/Science/JWST/MRS/GQLup/Results_new/',
              science_path = "/Users/gcugno/Science/JWST/MRS/GQLup/Raw_data/GQLup/",
              refs_path = "/Users/gcugno/Science/JWST/MRS/GQLup/Raw_data/",
              refs_names = ['1640_10'],
              psf_name = '1536_22',
              band = band)
    
    gqlup_red.prepare_cubes(size=size_bands[band])

    gqlup_red.PSFsub(mask = 0.5)

    gqlup_red.SNR(b_sep_lit = 0.708,
                b_pa_lit = 279,
                r_in_FWHM=0.3)

    gqlup_red.Contrast_spectrum()
    
    gqlup_red.Extract_spectrum(spectrum_path = "/Users/gcugno/Science/JWST/MRS/GQLup/Raw_data/spectrum_1536_obs22",
                                outlier_thres = thres_out[band])
