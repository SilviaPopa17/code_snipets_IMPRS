import numpy as np
import pyneb as pn
import pickle
import ai4neb
import time 


print('Load intensity map')
intensity_map = pickle.load(open('Stored_data/intensity_map.pkl', 'rb'))  #load intensities of all lines

#indexing of lines
#intensity_map --->'[Halpha]', '[Hbeta]' ,'[NII]', '[NII]', '[SII]', '[SII]', '[ClIII]', '[ClIII]', '[SIII]', '[SIII]']   
#                   6562,82,    4861,      5754.64, 6548.04, 6730.82, 6716.44,  5517.71,    5537.88,  6312.06, 9068.60]  



# choose atomic data for SII
pn.atomicData.setDataFile('s_ii_atom_IFF05.dat')
S2=pn.Atom("S",2)             

# load diagnostics 
diags = pn.Diagnostics(addAll = True)


# load specifications for ANN
diags.ANN_inst_kwargs = {'RM_type' : 'SK_SVM', 
                         'verbose' : True, 
                         'scaling' : True,
                         'use_log' : True
                                }
diags.ANN_init_kwargs = {'C' : 1000, 
                         'cache_size' : 200, 
                         'coef0' : 0.0, 
                         'degree' : 3, 
                         'epsilon' : 0.1, 
                         'gamma' : 'scale',
                         'kernel' : 'rbf', 
                         'max_iter' : -1, 
                         'shrinking' : True, 
                         'tol' : 0.00001}
diags.ANN_inst_kwargs = {'RM_type' : 'XGB', 
                         'verbose' : True, 
                         'scaling' : True,
                         'use_log' : True
                                }
diags.ANN_init_kwargs = {'n_estimators':1000,
                        'max_depth':6}
# Set up some parameters of the Regresor Method object
diags.ANN_inst_kwargs = {'RM_type' : 'SK_ANN', 
                         'verbose' : True, 
                         'scaling' : True,
                         'use_log' : True,
                         'random_seed' : 43
                                }
# Set up some parameters of the Analogic Neural Network
diags.ANN_init_kwargs = {'hidden_layer_sizes' : (10, 30, 10),
                         'activation' : 'tanh',
                         'solver' : 'lbfgs', 
                         'max_iter' : 20000} #
diags.ANN_n_tem = 30
diags.ANN_n_den = 30

#---------------------------------------------------

tem_ns = np.zeros((295,353))
tem_nc = np.zeros((295,353))

den_ns = np.zeros((295,353))
den_nc = np.zeros((295,353))



label     = ['Halpha', 'Hbeta','[NII]', '[NII]', '[SII]',  '[SII]', '[ClIII]', '[ClIII]', '[SIII]', '[SIII]']   
line_list = [6562.8,    4861.32, 5754.64, 6548.04, 6730.82, 6716.44,  5517.71,   5537.88,  6312.06,  9068.60]  

# cube ranges
n=1
m=294
q=1
r=353

print('Start tem and den computation')
start = time.time()
for i in range(n,m):
    for j in range(q,r):
        
        # intensity ratios
        rN2 = intensity_map[2,i,j]/intensity_map[3,i,j] #ratio of [NII] 5755/6548
        rS2 = intensity_map[4,i,j]/intensity_map[5,i,j] #ratio of [SII] 6731/6716
        rCl3 = intensity_map[7,i,j]/intensity_map[6,i,j] #ratio of [ClIII] 5538/5518
       
    
        # diagnostics with T from NII
        # put conditions so that PyNeb does not run if any of the values are nan values as it cannot handle those 
        
        if (rN2 is not np.ma.masked) and (rS2 is not np.ma.masked): 
            tem_ns[i,j], den_ns[i,j] = diags.getCrossTemDen('[NII] 5755/6548', '[SII] 6731/6716', rN2, rS2,  use_ANN=True, end_den=1e6, start_tem=3e3,limit_res=False) 
        if (rN2 is not np.ma.masked) and (rCl3 is not np.ma.masked): 
            tem_nc[i,j], den_nc[i,j] = diags.getCrossTemDen('[NII] 5755/6548', '[ClIII] 5538/5518', rN2, rCl3,  use_ANN=True, end_den=1e6, start_tem=3e3,limit_res=False) 
         
print('Done computing tem and den')   

end = time.time()
print('time for ANN method: {:1f} s.'.format(end - start))

pickle.dump(tem_ns, open('tem_ns.pkl', 'wb'))
pickle.dump(tem_nc, open('tem_nc.pkl', 'wb'))

pickle.dump(den_ns, open('den_ns.pkl', 'wb'))
pickle.dump(den_nc, open('den_nc.pkl', 'wb'))

print('Done saving data')
