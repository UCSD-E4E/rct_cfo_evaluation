'''CFO Estimator
'''
import numpy as np

def cfo_est (signal, f_s): # R = m
    # return np.angle(np.sum([signal[m+1]*np.conj(signal[m]) for m in range(len(signal)-1)])) * f_s / (2*np.pi) # single line implementation!
    fsum=0
    for m in range(len(signal)-1):    
        fsum += signal[m+1]*np.conj(signal[m])
    return np.angle(fsum) * f_s / (2*np.pi)


#     U = int(len(signal)/R)  
#     fsum=0
#     for u in range(U):
#         phi= 0
# #         autocorrelation
#         for m in range(u*R, u*R+R-1):
#             phi += signal[m+1]*np.conj(signal[m])
#         fsum += np.angle(phi)
#     return fsum * f_s / (2*np.pi * U)

#     U = int(len(signal)/R)  
#     fsum=0
#     for u in range(1,U):
#         phi= 0
# #         autocorrelation
#         for m in range(u*R, u*R+R):
#             phi += signal[m-R]*np.conj(signal[m])
#         fsum += np.angle(phi)
#     return fsum * f_s / (2*np.pi * (U-1))