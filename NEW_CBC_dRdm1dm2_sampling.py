import numpy as np
from astropy import constants
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats, integrate, interpolate
from scipy.ndimage import gaussian_filter
import popsummary
import glob, os
from scipy.interpolate import RegularGridInterpolator
import pandas as pd


def normal(x, mu, sigma):
    N = np.exp(-0.5*(((x-mu)/sigma)**2))/np.sqrt(2*np.pi*(sigma**2))
    return N

def l_M(x, M, NU):
    ratio = (x/M)**NU
    lm = 1/(1. + ratio)
    return lm

def extract_hyperparameters_from_popsummary(popsummary_obj, hyperparameter_list):

    '''
    Extracts hyperparameters from a popsummary object, fixes the annoying shape issues and creates a dictionary of 1D arrays
    '''

    hyperparameter_dict = {key:popsummary_obj.get_hyperparameter_samples(hyperparameters=[key]) for key in hyperparameter_list}

    return {key:hyperparameter_dict[key].reshape((hyperparameter_dict[key].size, )) for key in hyperparameter_dict.keys()}


### Load GWTC-5 data on full CBC population
fullpop_run_file = './production_1_mass_NotchFilterBinnedPairingMassDistribution_redshift_powerlaw_mag_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_popsummary.h5'
# fullpop_run_file = '/home/pracchia/GWTC5_rates/popsummary_files/production_1_mass_NotchFilterBinnedPairingMassDistribution_redshift_powerlaw_mag_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_popsummary.h5'
fullpop_result = popsummary.popresult.PopulationResult(fname=fullpop_run_file)
fullpop_hyperparameters = fullpop_result.get_hyperparameter_samples() # Posterior samples of the model hyperparameters

mass, R0 = fullpop_result.get_rates_on_grids('primary_mass')
m = mass[0]


m_BHmax = fullpop_hyperparameters.T[2]
m_BHmin = fullpop_hyperparameters.T[3]
m_NSmax = fullpop_hyperparameters.T[4]
m_NSmin = fullpop_hyperparameters.T[5]
m_UMGmax = fullpop_hyperparameters.T[6]
m_UMGmin = fullpop_hyperparameters.T[7]
alpha_1 = fullpop_hyperparameters.T[8]
alpha_2 = fullpop_hyperparameters.T[9]
alpha_dip = fullpop_hyperparameters.T[11]
beta_pair_1 = fullpop_hyperparameters.T[14]
beta_pair_2 = fullpop_hyperparameters.T[15]
A = fullpop_hyperparameters.T[0]
A2 = fullpop_hyperparameters.T[1]
mu1 = fullpop_hyperparameters.T[265]
mu2 = fullpop_hyperparameters.T[266]
sig1 = fullpop_hyperparameters.T[279]
sig2 = fullpop_hyperparameters.T[280]
rate = fullpop_hyperparameters.T[276]
mbreak = 5.
nu_Mns_min = 50.
nu_Mns_max = 50.
nu_Mbh_min = 50.
nu_Mbh_max = fullpop_hyperparameters.T[274]
nu_Mumg_min = 30.
nu_Mumg_max = 30.
c1 = fullpop_hyperparameters.T[263]
c2 = fullpop_hyperparameters.T[264]

# CBC_rate_tot = fullpop_hyperparameters.T[338]
# total_CBC_rate = fullpop_hyperparameters.T[338]


print('')
print('Computing one-dimensional mass distributions from hyperparameters (FullPop4.0 model)')

p_m_lambda = []
dip_factor = np.zeros_like(m)

for i in tqdm(range(len(rate))):
    # Normal distributions
    N1 = normal(m, mu1[i], sig1[i])
    N1 /= np.trapezoid(N1,m)
    N2 = normal(m, mu2[i], sig2[i])
    N2 /= np.trapezoid(N2,m)
    normal_factor = 1. + c1[i]*N1 + c2[i]*N2
    
    # High-pass, low-pass and notch functions
    l_m_BHmax = l_M(m, m_BHmax[i], nu_Mbh_max[i])
    h_m_NSmin = 1. - l_M(m, m_NSmin[i], nu_Mns_min)
    n1_m_NSmax_mBHmin = 1. - A[i]*l_M(m, m_NSmax[i], nu_Mns_max)*(1. - l_M(m, m_BHmin[i], nu_Mbh_min))
    n2_m_UMG = 1. - A2[i]*l_M(m, m_UMGmin[i], nu_Mumg_min)*(1. - l_M(m, m_UMGmax[i], nu_Mumg_max))
    
    # Factors for the mass gap dip
    dip_factor[m<m_NSmax[i]] = m[m<m_NSmax[i]]**alpha_1[i]
    dip_factor[m>=m_NSmax[i]] = (m[m>=m_NSmax[i]]**alpha_dip[i])*(m_NSmax[i]**(alpha_1[i]-alpha_dip[i]))
    dip_factor[m>=m_BHmin[i]] = (m[m>=m_BHmin[i]]**alpha_2[i])*(m_NSmax[i]**(alpha_1[i]-alpha_dip[i]))*(m_BHmin[i]**(alpha_dip[i]-alpha_2[i]))

    # Combine everything
    pi_m = normal_factor*l_m_BHmax*h_m_NSmin*n1_m_NSmax_mBHmin*n2_m_UMG*dip_factor
    pi_m /= np.trapezoid(pi_m, m)
    p_m_lambda.append(pi_m) 


print('')
print('Computing dR/dm1m2 samples')

f_m1_m2_beta = []
f_m1m2beta = np.zeros((len(m),len(m)))

dr_dm1dm2_samples = []

m1g = m.reshape([len(m),1])
m2g = m.reshape([1, len(m)])
theta_m1_m2 = np.heaviside(m1g-m2g,1)

m_less_break = m[m<=mbreak]
m_great_break = m[m>mbreak]
m2_less_break_g = m_less_break.reshape([1,len(m_less_break)])
m2_great_break_g = m_great_break.reshape([1,len(m_great_break)])

for k in tqdm(range(len(rate))):
    # p_m_lambda grids
    p_m1_lambda_g = p_m_lambda[k].reshape([len(m), 1])
    p_m2_lambda_g = p_m_lambda[k].reshape([1, len(m)])
    # Pairing function
    f_m1m2beta[:, m<=mbreak] = (m2_less_break_g/m1g)**beta_pair_1[k]
    f_m1m2beta[:, m>mbreak] = (m2_great_break_g/m1g)**beta_pair_2[k]
    # Combining
    drdm1dm2 = p_m1_lambda_g*p_m2_lambda_g*f_m1m2beta*theta_m1_m2
    drdm1dm2 /= np.trapezoid(np.trapezoid(drdm1dm2, m, axis=1), m, axis=0)
    drdm1dm2 *= rate[k]
    dr_dm1dm2_samples.append(drdm1dm2)


print('')
print('Saving data in running folder')
savepath = './O4b_joint_CBC_rates.npy'
np.save(savepath, dr_dm1dm2_samples)
