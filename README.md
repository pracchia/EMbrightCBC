# EMbrightCBC

Order in which launch the scripts:

1) `NEW_CBC_dRdm1dm2_sampling.py` gives the sampled 2d rate vs m1m2 curves. To run it download the GWTC-5 population study results from https://zenodo.org/records/20292639 (`popsummary_files.tar.gz`) extract it and copy the `popsummary_files/production_1_mass_NotchFilterBinnedPairingMassDistribution_redshift_powerlaw_mag_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_popsummary.h5` file in the working directory.

2) `continuous_embright_s22.py` estimates the EM-bright local rate densities for BNS and NSBH mergers using the EoS description of [Salafia et al. 2022](https://www.aanda.org/articles/aa/abs/2022/10/aa43260-22/aa43260-22.html).