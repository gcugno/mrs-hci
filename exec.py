from reduction import MRS_HCI


gqlup_red = MRS_HCI(output_dir = '/Users/gcugno/Science/JWST/MRS/GQLup/Results_new/',
              science_path = "/Users/gcugno/Science/JWST/MRS/GQLup/GQLup_MIRI/GQLUP/cubes_processed_150923/")


gqlup_red.prepare_cubes()

gqlup_red.PSFsub(pca_number = 7,
           mask = 0.5)

gqlup_red.SNR(b_sep_lit = 0.708,
        b_pa_lit = 279)

gqlup_red.Contrast_spectrum()
