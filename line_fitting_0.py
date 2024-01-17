import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import astropy.units as u
from astropy.table import Table
from mpdaf.obj import Cube
from mpdaf.obj import deg2sexa
import pandas as pd
import pyneb as pn
from astropy import visualization as viz
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.patches as patches
from scipy.interpolate import interp1d
from IPython.core.display import display, HTML
from astropy import visualization as viz
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.colors as colors
import matplotlib.colors as mcolors
import pickle



#important diagnostic lines; include also H_alpha and H_beta
label = ['Halpha', 'Hbeta','[NII]', '[NII]', '[SII]', '[SII]', '[ClIII]', '[ClIII]', '[SIII]', '[SIII]']   
line_list = [6562.8, 4861.32, 5754.64, 6548.04, 6730.82, 6716.44,  5517.71,5537.88,  6312.06, 9068.60]  


#RANGE SETTINGS

#----------------------------------------------------

# line fit range
d_l = [9,  10, 6.5, 7, 6, 6, 5.5,   6.5, 6,   10] #adjust min and max of fit
d_r = [10, 9.5, 7, 5, 7, 5,  6.5,  6.5, 6, 10]
   
# continuum range and width
d_cont_l = [50, 45, 28, 23, 9, 25, 28, 15, 7, 21]
delta_l  = [25, 25, 18, 15, 5, 15, 18, 10, 3, 10]
d_cont_r = [40, 35, 15, 45, 15, 5, 5.5, 8, 10, 12]
delta_r  = [20, 15, 15, 15, 10, 5, 8, 15, 15, 10]
   
# limes of the plot    
xlim_l = [60, 50, 15, 25, 20, 17, 20, 20, 20, 27 ] 
xlim_r = [60, 70, 15, 60, 21, 20, 30, 20, 20, 30 ]

#--------------------------------------------------

cube = Cube('../MUSE_data/muse-hr-fullcube-rebin05x05.fits')
print('Read cube')

# load corrected and normalized spectrum
cube_cor = pickle.load(open('Stored_data/cube_cor.pkl', 'rb'))


intensity_map = np.ma.masked_array(np.empty((5, 295, 353)), mask = True) #intensities; acces intensity for individual lines with intensity[i]
S_N_map = np.ma.masked_array(np.empty((len(label), 295, 353)), mask = True) #S/N for every line in each pixel
fwhm_map =  np.ma.masked_array(np.empty((len(label), 295, 353)), mask = True) #fwhm for every line in each pixel 
intensity_err_map = np.ma.masked_array(np.empty((len(label), 295, 353)), mask = True)  #err associated with intenstiy determination

n= 1
m= 294
q= 1
r= 352

import datetime
now = datetime.datetime.now()
print(now)

print('Start fitting')
for ii, line in enumerate(line_list[0:5]):
    print('Line', line)
    for i in range(n,m):
        for j in range(q,r):

            sp = cube[:, i, j]  #entire wavelength spectrum for pixel (i,j)
            sp.data = cube_cor[:, i, j] #update fluxes with intensities 
            
            if sp.data[0] is np.ma.masked: #ignore masked pixels, values remains masked - np.ma.masked 
                #print('pixel is masked so skip to next iteration')
                continue 
            
            # ranges of line fit
            lmin = line - d_l[ii]
            lmax = line + d_r[ii]  
            
            #chose region only around line
            sp_l = sp.subspec(lmin, lmax, unit=u.angstrom) 
      
            #CONTINUUM FITTING AND SUBTRACTION
            
            #limits of continuum
            cont_start_l = line - d_cont_l[ii]
            cont_end_l = cont_start_l + delta_l[ii]
            cont_start_r = line + d_cont_r[ii]
            cont_end_r = cont_start_r + delta_r[ii]
        
            #selection of continuum (left and right)
            cont = list(sp.subspec(cont_start_l, cont_end_l).data) + list(sp.subspec(cont_start_r, cont_end_r).data)     
            lam_cont = list(sp.subspec(cont_start_l, cont_end_l).wave.coord()) + list(sp.subspec(cont_start_r, cont_end_r).wave.coord())
    
            #do linear fit 
            coef= np.polyfit(lam_cont, cont, 1)
            fit = np.polyval(coef, lam_cont)
            interp_func = interp1d(lam_cont, fit, kind='linear') #continuum fit interpolated
        
            #do cont substraction
            flux_cont_sub = sp_l.data - interp_func(sp_l.wave.coord()) #interpolate over lam points of line spectrum
            
            #update the spectrum with cont subtracted flux values
            sp_l.data = flux_cont_sub
                   
            #S/N CUT 
            
            S = np.max(sp_l.subspec(line-2, line+2).data) #take S as max flux; be precise in the range you look at; it can be that the max flux is at the sides in a bad line; so pick only small range around center
            N = np.std(cont) #take N as standard deviation of continuum; without the masked part
        
           
            if S/N < 3:
                continue
      
            S_N_map[ii, i, j] = S/N

           
            #FITTING
            try:
                line_fit = sp_l.gauss_fit(lmin, lmax, plot=False)
                intensity_map[ii, i, j] = line_fit.flux #intensity (corrected flux)
                fwhm_map[ii,i,j] = line_fit.fwhm
                intensity_err_map[ii, i, j] = line_fit.err_flux/line_fit.flux*100
            except ValueError as e:
                print(f"Error occurent:{e}", i, j)
                continue   
                
print('Done fitting')
now = datetime.datetime.now()
print(now)

# SAVE ARRAY

pickle.dump(intensity_map, open('Stored_data/intensity_map_0.pkl', 'wb'))   #store intensities 
pickle.dump(S_N_map, open('Stored_data/S_N_map_0.pkl', 'wb'))               #store S/N 
pickle.dump(fwhm_map, open('Stored_data/fwhm_map_0.pkl', 'wb'))            
pickle.dump(intensity_err_map, open('Stored_data/intensity_err_map_0.pkl', 'wb'))  

print('Done saving')
