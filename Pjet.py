import numpy as np
import os
import pathlib
here = pathlib.Path(__file__).parent.resolve()

# GW170817 component masses from high-spin-prior analysis
m1_17,m2_17 = np.load(os.path.join(here,'data/GW170817_m1m2_posterior_samples_HS.npy'))
m17=(m1_17+m2_17)
c17=np.linspace(0.,1.,len(m17))

# PSRJ0740+6620 mass according to Fonseca et al.
mpsr = np.sort(np.random.normal(2.08,0.07,3000))
cpsr = np.linspace(0.,1.,len(mpsr))

def Mdyn_BNS_KF20(m1,m2,C1,C2,a=-9.3335,b=114.17,c=-337.56,n=1.5465):
    """
    BNS dynamical ejecta mass according to Kruger & Foucart 2020
    
    Input: 
    - m1,m2: component gravitational masses in Msun
    - C1,C2: compactnesses
    
    Output: Mdyn
    - Mdyn: dynamical ejecta mass in solar masses
    """
    return 1e-3*(m1*(a/C1+b*(m2/m1)**n+c*C1) + m2*(a/C2+b*(m1/m2)**n+c*C2))
    
def Mdisk_BNS_KF20(m2,C2,a=-8.1324,c=1.4820,d=1.7784):
    """
    BNS disk mass according to Kruger & Foucart 2020
    
    Input: 
    - m2: secondary gravitational mass in Msun
    - C2: compactness of the secondary
    
    Output: Mdisk
    - Mdisk: accretion disk mass in solar masses
    """
    return m2*np.maximum(a*C2+c,0.)**d

def hatMrem_NSBH_F18(m1,m2,C2,chi1,a=0.406,b=0.139,g=0.255,d=1.761):
    """
    NSBH remnant mass divided by NS baryonic mass according to Foucart et al. 2018
    
    Input: 
    - m1,m2: component gravitational masses in Msun
    - C2: compactness of the neutron star
    - chi1: spin parameter of black hole (assumed aligned with orbital angular momentum)
    
    Output: Mrem/mb2
    - Mrem/mb2: ratio of the mass outside the horizon after merger to
                the baryonic mass of the NS
    """
    Q = m1/m2
    eta = Q/(1.+Q)**2
    Z1 = 1.+(1.-chi1**2)**(1./3.)*((1.+chi1**2)**(1./3.)+(1.-chi1**2)**(1./3.))
    Z2 = np.sqrt(3.*chi1**2+Z1**2)
    Risco = 3.+Z2-np.sign(chi1)*np.sqrt((3.-Z1)*(3.+Z1+2.*Z2))
    return np.maximum(a*(1.-2.*C2)/eta**(1./3.)-b*Risco*C2/eta+g,0.)**d

def mb_from_m_and_C(m,C):
    """
    baryonic mass given a gravitational mass and a compactness, based
    on the binding energy approximation of Lattimer & Prakhash 2001
    """
    return m*(1.+0.6*C/(1.-0.5*C))

def Mdyn_NSBH_K20(m1,m2,C2,chi1,a1=0.007116,a2=0.001436,a4=-0.02762,n1=0.8636,n2=1.6840):
    """
    NSBH dynamical ejecta mass according to Kruger & Foucart 2020
    
    Input: 
    - m1,m2: component gravitational masses in Msun
    - C2: compactness of the neutron star
    - chi1: spin parameter of black hole (assumed aligned with orbital angular momentum)
    
    Output: Mdyn
    - Mdyn: dynamical ejecta mass in solar masses
    """
    
    Q = m1/m2
    Z1 = 1.+(1.-chi1**2)**(1./3.)*((1.+chi1**2)**(1./3.)+(1.-chi1**2)**(1./3.))
    Z2 = np.sqrt(3.*chi1**2+Z1**2)
    Risco = 3.+Z2-np.sign(chi1)*np.sqrt((3.-Z1)*(3.+Z1+2.*Z2))
    mb2 = m2*(1.+0.6*C2/(1.-0.5*C2))
    return mb2*np.maximum(a1*Q**n1*(1.-2.*C2)-a2*Q**n2*Risco*C2+a4,0.)


def p_MTOV_RNS_ad(MTOV,RNS,ad):
    """
    Conditional probability of M_TOV given R_NS and alpha_d
    
    Input:
    - MTOV: Tolman-Oppenheimer-Volkoff mass in Msun
    - RNS: neutron star radius in km
    - ad: alpha_d nuisance parameter (ratio of true BNS disk mass to 
          the mass predicted by Krueger & Foucart 2020)
    
    Output:
    - p(M_TOV | R_NS, alpha_d) in Msun^-1
    """
    
    # compute the GW170817 remnant mass, following S22    
    nu17 = m1_17*m2_17/(m1_17+m2_17)**2 # symmetric mass ratio
    C17 = 1.48*m17/RNS                  # "total compactness"
    c1_17 = 1.48*m1_17/RNS              # primary compactness
    c2_17 = 1.48*m2_17/RNS              # secondary compactness
    mgw17 = 0.25*nu17*C17*m17           # mass in GW
    md17 = xid*Mdisk_BNS_KF20(m2_17,c2_17) # disk mass (depends on RNS!)
    me17 = Mdyn_BNS_KF20(m1_17,m2_17,c1_17,c2_17) # ejecta mass (depends on RNS!)
    mrem_17 = m17-mgw17-md17-me17       # remnant mass
    
    # MTOV must be between the maximum pulsar mass and the GW170817 remnant mass divided by 1.2
    mtov = np.linspace(1.8,2.6,100)
    pmtov = np.interp(mtov,mpsr,cpsr)*(1.-np.interp(mtov,np.sort(mrem_17)/1.2,c17))
    pMTOV = np.nan_to_num(np.interp(MTOV,mpsr,cpsr)*(1.-np.interp(MTOV,np.sort(mrem_17)/1.2,c17))/np.trapezoid(pmtov,mtov))
        
    return pMTOV 

if __name__=='__main__':
    
    import matplotlib.pyplot as plt

    plt.rcParams['font.family']='serif'
    plt.rcParams['text.usetex']=True
    
    Nsamples = int(3e3)
    
    Mdmins = [1e-3,1e-2]
    chi1 = np.array([0.,0.2,0.6])
    MTOV = np.random.uniform(1.8,2.6,Nsamples)
    RNS = np.random.normal(12.45,0.65,Nsamples)
    ad = np.exp(np.random.normal(-0.11,0.47,Nsamples))
    bd = np.exp(np.random.normal(-0.0020,0.20,Nsamples))
    
    m1 = np.geomspace(1.,10.,200)
    m2 = np.geomspace(1.,2.2,151)
    jet = np.zeros([Nsamples,len(chi1),len(m1),len(m2)])
    
    plt.figure(figsize=(12.,7))
        
    for k,Mdmin in enumerate(Mdmins):
    
        for i in range(Nsamples):
            C1 = 1.48*m1/RNS[i]
            C2 = 1.48*m2/RNS[i]
            MdBNS = ad[i]*Mdisk_BNS_KF20(m2,C2)
            MejBNS = Mdyn_BNS_KF20(m1[:,None],m2[None,:],C1[:,None],C2[None,:])
            M = (m1[:,None]+m2[None,:])
            nu = m1[:,None]*m2[None,:]/M**2
            C = 1.48*M/RNS[i]                 
            Mgw = 0.25*nu*C*M          
            MremBNS = M-Mgw-MdBNS[None,:]-MejBNS
            
            for j in range(len(chi1)):
                hatMrem = hatMrem_NSBH_F18(m1[:,None],m2[None,:],C2[None,:],chi1[j])
                MdNSBH = mb_from_m_and_C(m2[None,:],C2[None,:])*(bd[i]*hatMrem)
            
                jet[i,j] = p_MTOV_RNS_xid(MTOV[i],RNS[i],ad[i])*(m1[:,None]>=m2[None,:])*np.where((m1[:,None]<=MTOV[i]),(MdBNS>Mdmin)[None,:]&(MremBNS>(1.2*MTOV[i])),(MdNSBH>Mdmin)&(m2[None,:]<MTOV[i]))
        
        
        pjet = np.mean(jet,axis=0)
        
        for j in range(len(chi1)):
            plt.subplot(2,3,3*k+j+1)
            plt.semilogx()
            plt.xlim(1.,10.)
            plt.ylim(1.,2.2)
            plt.tick_params(which='both',direction='in',top=True,right=True)
            plt.xticks([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.],['1','2','3','4','5','6','7','8','9',''])
            
            plt.title(r'$\chi_1=%.1f$, $M_\mathrm{disk,min}=%.3f\,\mathrm{M_\odot}$'%(chi1[j],Mdmin))
            cs = plt.contour(m1,m2,pjet[j].T,levels=[0.01,0.25,0.5,0.75,0.99],linewidths=2.,vmax=1.02)
            if j==2:
                plt.clabel(cs,fmt={0.01:r'$1\%$',0.25:r'$25\%$',0.5:r'$50\%$',0.75:r'$75\%$',0.99:r'$99\%$',},manual=[(7.1,1.9),(5.8,1.9),(6.8,1.68),(8.45,1.4),(6.1,1.6)])
        
            if k>0:
                plt.xlabel(r'$m_1/\mathrm{M_\odot}$')
            if j==0:
                plt.ylabel(r'$m_2/\mathrm{M_\odot}$')
    
    
    plt.show()    
         
    
