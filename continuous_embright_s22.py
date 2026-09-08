import numpy as np
from astropy import constants
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import interpolate
import popsummary
from scipy.interpolate import RegularGridInterpolator

G = constants.G
c = constants.c
m_sun = constants.M_sun
A = (G*m_sun/c/c).to('km').value # Constants in kilometers

### Globals
res = 100 # Grid resolution
m_ns_min = 1.
m_bh_max = 30.
m_split = 2.5
chi_min = -1.
chi_max = 1.
r_ns_min = 9.
r_ns_max = 15.
m_rem_min = 5e-4

def R_isco(chi_BH):
    '''
    chi_BH: effective black hole spin
    Returns the ISCO radius normalized to the BH mass
    '''
    Z1 = 1. + ((1. - chi_BH**2.)**(1./3.))*((1. + chi_BH)**(1./3.) + (1. - chi_BH)**(1./3.))
    Z2 = np.sqrt(3*chi_BH**2. + Z1**2.)
    R_isco = 3. + Z2 - np.sign(chi_BH)*np.sqrt((3. - Z1)*(3 + Z1 + 2*Z2))
    return R_isco

def M_rem(M_BH, chi_BH, M_NS, C_NS, a=0.406, b=0.139, g=0.255, d=1.761):
    '''
    M_BH: black hole mass
    chi_BH: effective black hole spin
    M_NS: neutron star mass
    C_NS: neutron star compaction
    Returns the barionic remnant mass
    All masses are expressed in solar masses
    '''
    Q = M_BH/M_NS # Mass ratio
    eta = Q/((1+Q)**2.) # Symmetric mass ratio
    equat_M_rem = a*(1-(2*C_NS))/(eta**(1./3.)) - b*R_isco(chi_BH)*C_NS/eta + g
    if np.isscalar(equat_M_rem):
        M_rem = np.max((equat_M_rem,0))**d
    else:
        equat_M_rem[equat_M_rem<=0.] = 0.
        M_rem = equat_M_rem**d
    return M_rem*M_NS


def M_rem_R_NS(M_BH, chi_BH, M_NS, R_NS, a=0.406, b=0.139, g=0.255, d=1.761):
    '''
    M_BH: black hole mass
    chi_BH: effective black hole spin
    M_NS: neutron star mass
    R_NS: neutron star radius in kilometers
    Returns the barionic remnant mass
    All masses are expressed in solar masses
    '''
    Q = M_BH/M_NS # Mass ratio
    eta = Q/((1+Q)**2.) # Symmetric mass ratio
    # C_NS = G*M_NS/R_NS/c/c = (G*M_sun/c/c)*m_ns/R = A*m_ns/R
    C_NS = A*M_NS/R_NS
    equat_M_rem = a*(1-(2*C_NS))/(eta**(1./3.)) - b*R_isco(chi_BH)*C_NS/eta + g
    if np.isscalar(equat_M_rem):
        M_rem = np.max((equat_M_rem,0))**d
    else:
        equat_M_rem[equat_M_rem<=0.] = 0.
        M_rem = equat_M_rem**d
    return M_rem*M_NS


def M_disk_BNS_R_NS(M_NS, R_NS, a_bns=-8.1324, c_bns=1.4820, d_bns=1.7784):
    '''
    M_NS: neutron star mass
    R_NS: neutron star radius in kilometers
    Returns the barionic remnant mass
    All masses are expressed in solar masses
    '''
    C_NS = A*M_NS/R_NS
    disk_mass_fit = np.nan_to_num((a_bns*C_NS + c_bns)**d_bns)
    # disk_mass_fit[disk_mass_fit<5e-4] = 5e-4
    Mdisk = M_NS*disk_mass_fit
    Mdisk[Mdisk < m_rem_min] = 0.
    return Mdisk


def M_ejecta_BNS_R_NS(M_1, M_2, R_1, R_2, a_ej=-9.3335, b_ej=114.17, c_ej=-337.56, n_ej=1.5465):
    '''
    M_1: primary neutron star mass
    M_2: secondary neutron star mass
    R_1: primary neutron star radius in kilometers
    R_2: secondary neutron star radius in kilometers
    Returns the barionic remnant mass
    All masses are expressed in solar masses
    '''
    C_1 = A*M_1/R_1
    C_2 = A*M_2/R_2
    ejecta_mass_fit_1 = ((a_ej/C_1) + b_ej*(M_2**n_ej)/(M_1**n_ej) + c_ej*C_1)*M_1
    ejecta_mass_fit_2 = ((a_ej/C_2) + b_ej*(M_1**n_ej)/(M_2**n_ej) + c_ej*C_2)*M_2
    Mej = ejecta_mass_fit_1 + ejecta_mass_fit_2
    Mej[Mej<0.] = 0.
    return Mej*1e-3

### Mass array from GWTC-5 CBC results
pdb_file = '/home/pracchia/GWTC5_rates/popsummary_files/production_1_mass_NotchFilterBinnedPairingMassDistribution_redshift_powerlaw_mag_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_popsummary.h5'
pdb_result = popsummary.popresult.PopulationResult(fname=pdb_file)
MASS1, drdm1 = pdb_result.get_rates_on_grids('primary_mass')
m = MASS1[0]
del pdb_result
del MASS1
del drdm1

### Chi_z distribution
chiz, p_chiz = np.load('chi_costheta_hyp_smooth.npy')
p_chiz_interp = interpolate.interp1d(chiz,p_chiz)


### Load GWTC-5 dr_dm1dm2 samples data
savepath = './O4b_joint_CBC_rates.npy'
dr_dm1dm2_samples = np.load(savepath)

print('')
print('Interpolating dR/dm1dm2 sampled distributions')
print('')

interpolated_dr_dm1dm2_samples = [None]*len(dr_dm1dm2_samples)
for i, sample in enumerate(tqdm(dr_dm1dm2_samples)):
    interpolated_dr_dm1dm2_samples[i] = RegularGridInterpolator(points=(m,m),values=np.nan_to_num(sample),bounds_error=False) 

del dr_dm1dm2_samples

print('Loading R14, MTOV and xi_d samples')
print('')
samples_r14_mtov_xid = np.load('./data/samples_r14_mtov_xid.npy')


print('Computing EMbright probabilities')
print('')

m1_1 = np.linspace(m_ns_min, m_split, res)
m1_2 = np.linspace(m_split, m_bh_max, res)
m2 = np.linspace(m_ns_min, m_split, res)

m1_mould = [m1_1, m1_2]
m1 = np.array(m1_mould).reshape(len(m1_1)+len(m1_2))
m1 = np.delete(m1, len(m1_2))

# m1 = np.linspace(m_ns_min, m_bh_max, 10*res)
m1g = m1.reshape([len(m1),1])
m2g = m2.reshape([1,len(m2)])

chi = np.linspace(chi_min,chi_max,res)
chig = chi.reshape([1,1,len(chi)])

p_chi_m1 = p_chiz_interp(chi)
p_chi_m1 = p_chi_m1.reshape([1,1,len(chi)])

p_embright_m1m2_samples = []
m1_NSBH_i = []
m1_BNS_i = []
m2_BNSBH_i = []
p_embright_m1m2_nsbh_samples = []
p_embright_m1m2_bns_samples = []
p_embright_m1m2_interp_samples = []
test = []

for r14, mtov, xid in tqdm(samples_r14_mtov_xid):    
    m2_BNSBH = np.linspace(m_ns_min, mtov, res+2)
    m2_BNSBH_i.append(m2_BNSBH)
    
    # NSBH
    m1_NSBH = np.linspace(mtov, m_bh_max, res-1)
    m1_NSBH_i.append(m1_NSBH)
    
    m1_NSBHg = m1_NSBH.reshape([len(m1_NSBH),1,1])
    m2_NSBHg = m2_BNSBH.reshape([1,len(m2_BNSBH),1])
    
    remnant_mass = M_rem_R_NS(m1_NSBHg, chig, m2_NSBHg, r14)
    theta_Mrem = np.heaviside(remnant_mass - m_rem_min, 0)
    theta_Mtov = np.heaviside(mtov - m2_NSBHg, 1)
    theta_Mtov_m2 = np.heaviside(mtov - m2_NSBHg, 1)
    p_embright_m1m2_dchi = theta_Mrem*p_chi_m1*theta_Mtov*theta_Mtov_m2
    p_embright_m1m2_NSBH = np.trapezoid(p_embright_m1m2_dchi, chi, axis=2)
    p_embright_m1m2_nsbh_samples.append(p_embright_m1m2_NSBH)
    
    # BNS
    m1_BNS = np.linspace(m_ns_min, mtov, res-1)
    m1_BNS_i.append(m1_BNS)
    # m2_BNS = np.linspace(m_ns_min, mtov, res+2)
    
    m1_BNSg = m1_BNS.reshape([len(m1_BNS),1])
    m2_BNSg = m2_BNSBH.reshape([1,len(m2_BNSBH)])

    
    M = m1_BNSg + m2_BNSg
    nu = m1_BNSg*m2_BNSg/(M**2)

    mgw = 0.25*nu*A*M**2/r14
    md = M_disk_BNS_R_NS(m2_BNSg, r14)*xid
    mej = M_ejecta_BNS_R_NS(m1_BNSg, m2_BNSg, r14, r14)
    mrem = M - mgw - md - mej
    # Heaviside functions
    theta_Mtov_m1 = np.heaviside(mtov - m1_BNSg, 1)
    theta_Mtov_m2 = np.heaviside(mtov - m2_BNSg, 1)
    theta_m1m2 = np.heaviside(m1_BNSg - m2_BNSg, 1)
    theta_Mdisk = np.heaviside(md - m_rem_min, 1)
    theta_BH_collapse = np.heaviside(mrem - 1.2*mtov, 1)
    # Combine
    p_embright_m1m2_BNS = theta_Mtov_m1*theta_Mtov_m2*theta_m1m2*theta_Mdisk*theta_BH_collapse
    p_embright_m1m2_bns_samples.append(p_embright_m1m2_BNS)

    # Join BNS and NSBH
    m1_mould = [m1_BNS, m1_NSBH]
    m1_stitch = np.array(m1_mould).reshape(len(m1_BNS)+len(m1_NSBH))
    m1_stitch = np.delete(m1_stitch, len(m1_BNS))
    
    p_embright_m1m2_mould = [p_embright_m1m2_BNS, p_embright_m1m2_NSBH]
    p_embright_m1m2_stitch = np.array(p_embright_m1m2_mould).reshape([len(m1_BNS)+len(m1_NSBH),len(m2_BNSBH)])
    p_embright_m1m2_stitch = np.delete(p_embright_m1m2_stitch, len(m1_BNS), axis=0)
    p_embright_m1m2_interp = interpolate.RegularGridInterpolator(points=(m1_stitch,m2_BNSBH),values=np.nan_to_num(p_embright_m1m2_stitch),bounds_error=False) 
    p_embright_m1m2_grid = np.nan_to_num(p_embright_m1m2_interp((m1g,m2g)))
    p_embright_m1m2_samples.append(p_embright_m1m2_grid)
    p_embright_m1m2_interp_samples.append(p_embright_m1m2_interp)
    
averaged_p_embright_m1m2 = sum(p_embright_m1m2_samples) / len(p_embright_m1m2_samples)


print('Computing EMbright rates')
print('')

R_EMbright_NSBH_s22 = np.zeros(len(interpolated_dr_dm1dm2_samples)*len(samples_r14_mtov_xid))
R_EMbright_BNS_s22 = np.zeros(len(interpolated_dr_dm1dm2_samples)*len(samples_r14_mtov_xid))
R_ratio_s22 = np.zeros(len(interpolated_dr_dm1dm2_samples)*len(samples_r14_mtov_xid))
NSBH_EMbright_fraction_s22 = np.zeros(len(interpolated_dr_dm1dm2_samples)*len(samples_r14_mtov_xid))
BNS_EMbright_fraction_s22 = np.zeros(len(interpolated_dr_dm1dm2_samples)*len(samples_r14_mtov_xid))
total_SGRB_s22 = np.zeros(len(interpolated_dr_dm1dm2_samples)*len(samples_r14_mtov_xid))
NSBH_percentage_s22 = np.zeros(len(interpolated_dr_dm1dm2_samples)*len(samples_r14_mtov_xid))


index = 0

for j in tqdm(range(len(samples_r14_mtov_xid))):
    m1_NSBH = m1_NSBH_i[j]
    m1_BNS = m1_BNS_i[j]
    m2_BNSBH = m2_BNSBH_i[j]
    m1_NSBHg = m1_NSBH.reshape(len(m1_NSBH),1)
    m2_NSBHg = m2_BNSBH.reshape(1,len(m2_BNSBH))
    m1_BNSg = m1_BNS.reshape(len(m1_BNS),1)
    m2_BNSg = m2_BNSBH.reshape(1,len(m2_BNSBH))
    p_embright_m1m2_NSBH = p_embright_m1m2_interp_samples[j]((m1_NSBHg, m2_NSBHg))
    p_embright_m1m2_BNS = p_embright_m1m2_interp_samples[j]((m1_BNSg, m2_BNSg))
    for i in range(len(interpolated_dr_dm1dm2_samples)):
        
        dr_dm1m2_NSBH_i = interpolated_dr_dm1dm2_samples[i]((m1_NSBHg, m2_NSBHg))
        dr_dm1m2_BNS_i = interpolated_dr_dm1dm2_samples[i]((m1_BNSg, m2_BNSg))
        
        R_NSBH = np.trapezoid(np.trapezoid(dr_dm1m2_NSBH_i,m2_BNSBH,axis=1), m1_NSBH, axis=0)
        R_BNS = np.trapezoid(np.trapezoid(dr_dm1m2_BNS_i,m2_BNSBH,axis=1), m1_BNS, axis=0)
        
        R_EMbright_NSBH_s22[index] = np.trapezoid(np.trapezoid(dr_dm1m2_NSBH_i*p_embright_m1m2_NSBH, m2_BNSBH, axis=1), m1_NSBH, axis=0)
        R_EMbright_BNS_s22[index] = np.trapezoid(np.trapezoid(dr_dm1m2_BNS_i*p_embright_m1m2_BNS, m2_BNSBH, axis=1), m1_BNS, axis=0)
        
        NSBH_EMbright_fraction_s22[index] = R_EMbright_NSBH_s22[index] / R_NSBH
        BNS_EMbright_fraction_s22[index] = R_EMbright_BNS_s22[index] / R_BNS
        
        R_ratio_s22[index] = R_EMbright_NSBH_s22[index] / R_EMbright_BNS_s22[index]
        total_SGRB_s22[index] = R_EMbright_BNS_s22[index] + R_EMbright_NSBH_s22[index]
        NSBH_percentage_s22[index] = 100 * R_EMbright_NSBH_s22[index] / total_SGRB_s22[index]
        
        index += 1


print('Saving data in running folder')
print('')

savepath = './data/m1_s22_continuous.npy'
np.save(savepath, m1)
savepath = './data/m2_s22_continuous.npy'
np.save(savepath, m2)
savepath = './data/averaged_p_embright_m1m2_s22_continuous.npy'
np.save(savepath, averaged_p_embright_m1m2)

savepath = './EMbright_rates/R_EMbright_NSBH_s22_continuous.npy'
np.save(savepath, R_EMbright_NSBH_s22)
savepath = './EMbright_rates/R_EMbright_BNS_s22_continuous.npy'
np.save(savepath, R_EMbright_BNS_s22)
savepath = './EMbright_rates/R_ratio_s22_continuous.npy'
np.save(savepath, R_ratio_s22)
savepath = './EMbright_rates/NSBH_EMbright_fraction_s22_continuous.npy'
np.save(savepath, NSBH_EMbright_fraction_s22)
savepath = './EMbright_rates/BNS_EMbright_fraction_s22_continuous.npy'
np.save(savepath, BNS_EMbright_fraction_s22)
savepath = './EMbright_rates/total_SGRB_s22_continuous.npy'
np.save(savepath, total_SGRB_s22)
savepath = './EMbright_rates/NSBH_percentage_s22_continuous.npy'
np.save(savepath, NSBH_percentage_s22)

print('Done')
print('')
