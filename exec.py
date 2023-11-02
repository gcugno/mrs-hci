from reduction import MRS_HCI


thres_out = {"1A":5, "1B":5, "1C":5,"2A":5, "2B":5, "2C":8}
size_bands = {"1A":9, "1B":9, "1C":9, "2A":9, "2B":8, "2C":8}


band = "1C"
gqlup_red = MRS_HCI(output_dir = '/Users/gcugno/Science/JWST/MRS/GQLup/Results_new/',
              science_path = "/Users/gcugno/Science/JWST/MRS/GQLup/Raw_data/GQLup/",
              refs_path = "/Users/gcugno/Science/JWST/MRS/GQLup/Raw_data/",
              refs_names = ['1050_3', '1050_5', '1050_9', '1524_1', '1524_2', '1524_3', '1524_4', '1524_6', '1524_7', '1524_8', '1524_17', '1536_22', '1536_23', '1538_1', '1538_2', '1538_3','1640_10'], # ALL
              #refs_names = ['1050_3', '1050_5', '1524_1', '1524_2', '1524_3', '1524_4', '1524_6', '1524_7', '1524_8', '1524_17', '1538_1', '1640_10'],
              #refs_names = ['1050_3', '1050_9', '1524_17', '1536_22', '1536_23', '1538_1', '1538_2', '1538_3', '1640_10'], # OLD VERSION
              psf_name = '1536_22',
              band = band)
              
              
#gqlup_red = MRS_HCI(output_dir = '/Users/gcugno/Science/JWST/MRS/GQLup/Results_new/',
#              science_path = "/Users/gcugno/Science/JWST/MRS/GQLup/Raw_data/GQLup/",
#              refs_path = "/Users/gcugno/Science/JWST/MRS/GQLup/GQLUP_MIRI_old!!_DELETE/",
#              refs_names = ['1050_3', '1050_9', '1524_17', '1536_22', '1536_23', '1536_24', '1538_1', '1538_2', '1538_3'],
#              psf_name = '1538_1',
#              band = "1A")
              
#gqlup_red = MRS_HCI(output_dir = '/Users/gcugno/Science/JWST/MRS/GQLup/Results_new/',
#              science_path = "/Users/gcugno/Science/JWST/MRS/GQLup/GQLUP_MIRI_old!!_DELETE/GQLUP/cubes_processed_150923/",
#              refs_path = "/Users/gcugno/Science/JWST/MRS/GQLup/GQLUP_MIRI_old!!_DELETE/",
#              refs_names = ['1050_3', '1050_9', '1524_17', '1536_22', '1536_23', '1536_24', '1538_1', '1538_2', '1538_3', 'VRYLUP'],
#              psf_name = '1538_1',
#              band = "1A")


gqlup_red.prepare_cubes(size=size_bands[band])

gqlup_red.PSFsub(pca_number = 14,
           mask = 0.4)

gqlup_red.SNR(b_sep_lit = 0.708,
        b_pa_lit = 279)

gqlup_red.Contrast_spectrum()

gqlup_red.Estimate_uncertainties(num_angles=10)

gqlup_red.Extract_spectrum(outlier_thres = thres_out[band])
