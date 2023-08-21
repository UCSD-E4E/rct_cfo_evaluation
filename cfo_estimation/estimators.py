'''CFO Estimator
'''
import numpy as np

def cfo_est (signal, f_s): # R = m
    # return np.angle(np.sum([signal[m+1]*np.conj(signal[m]) for m in range(len(signal)-1)])) * f_s / (2*np.pi) # single line implementation!
    fsum=0
    for m in range(len(signal)-1):    
        fsum += signal[m+1]*np.conj(signal[m])
    return np.angle(fsum) * f_s / (2*np.pi)

# def cfo_est_mle(self, signal): #MLE method

#         l_dash=int((len(signal)-1)/self.d)
#         w_mle=0
#         for mu in range(1, self.b+1):
#             sum_mle=0
#             p=np.arange(mu, self.b+1)
#             a1=l_dash+self.q*(p-mu)
#             a2=l_dash+self.q*(p-mu)+mu*self.q 
#             a4=np.where(a1<len(signal),a1,a1-len(signal))
#             a5=np.where(a2<len(signal),a2,a2-len(signal))
#             a6=np.where(a4<len(signal),a4,len(signal)-1)
#             a7=np.where(a5<len(signal),a5,len(signal)-1)
#             s1=np.take(signal, a6)
#             s2=np.take(signal, a7)
#             sum_mle=np.sum(self.autocorrelate_phi(s1,s2))
#             angle=np.arctan(np.imag(sum_mle)/np.real(sum_mle))
#             w_mle+=(angle/mu)

#         print("\nMLE method for CFO Est. Done. \n")

#         return w_mle/(self.q*self.psi) 