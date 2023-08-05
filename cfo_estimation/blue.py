'''BLUE CFO Estimator
'''
import numpy as np

def autocorrelate_phi(signal, signal_shift):
    return  signal * (np.real(signal_shift) - np.imag(signal_shift))

def offset_to_hz(x): 
    return  ((10**(-np.abs(x)))/20)*(10**5)

def cfo_est_blue(signal, f_s, m): #BLUE method
    total_n = len(signal)
    u = int(m)
    r = int(total_n/u)
    k = int(u/2)
    fsum = 0
    last_phiu = -1
    phiu = -1
    for i in range(k):
        mu = np.arange(i * r, total_n)
        mu2 = mu - i * r
        a4 = np.where(mu < len(signal), mu, mu - len(signal))
        a5 = np.where(mu2 < len(signal), mu2, mu2 - len(signal))
        s1 = np.take(signal, a4)
        s2 = np.take(signal, a5)
        phi_bu = np.sum(autocorrelate_phi(s1, s2)) / (total_n - i * r)
        if u >= 1:
            phiu = (np.arctan(np.imag(phi_bu) / np.real(phi_bu)) - np.arctan(np.imag(last_phiu) / np.real(last_phiu))) % (2 * np.pi)
        else:
            phiu = phi_bu
        last_phiu = phiu
        wu = ((3 * (u - i) * (u - i + 1)) - (k * (u - k))) / (k * (4 * k * k - 6 * u * k + 3 * u * u - 1))
        fsum += (wu * phiu)
    # print("\nBLUE Method CFO Est. Done. \n")
    return  offset_to_hz(1 / ((fsum * u * f_s) / (2 * np.pi)))

def unfinished_cfo_est (signal, f_s, R): # R = m
    N = len(signal)
    U = int(N/R)
    K = int(U/2)
    print(f'N={N},U={U},K={K}')
    phi_blue_curr = 0
    phi_blue_prev = 0
    phi_u = 0
    fsum=0
    for u in range(K):
        phi_blue_curr = 0
#         autocorrelation
        for m in range(u*R, N):
            phi_blue_curr += signal[m]*np.conj(signal[m-u*R])
        phi_blue_curr = phi_blue_curr / (N - u * R)
        if u >= 1:
            print(f'phi_blue_curr = {phi_blue_curr}')
            print(f'phi_blue_prev = {phi_blue_prev}')
            phi_u = np.angle(phi_blue_prev) - np.angle(phi_blue_curr)
            print(f'angle_curr = {np.angle(phi_blue_curr)}')
            print(f'angle_prev = {np.angle(phi_blue_prev)}')
            print(f'phi_u = {phi_u}')
            
            # wu = (3*(U - u) * (U - u + 1) - K * (U - K))/(K*(4*K*K - 6*U*K + 3*U*U - 1))    
            # fsum += wu*phi_u
            fsum += phi_u
        phi_blue_prev = phi_blue_curr
    # return fsum * U * f_s / (2*np.pi)
#     convert result from rad to Hz
    return fsum * f_s / (2*np.pi * (K-1))  


def cfo_est_aom_r(signal, b, b_dash, m, q, psi, p): #angle of mean with reuse 

    sum_aom_r=0
    k=np.arange(1,p+1)
    for i in range(1, b_dash+1):
        a1=q*i-q+k
        a2=q*i+k
        a4=np.where(a1<len(signal),a1,a1-len(signal))
        a5=np.where(a2<len(signal),a2,a2-len(signal))
        s1=np.take(signal, a4)
        s2=np.take(signal, a5)
        sum_aom_r+=(np.sum(autocorrelate_phi(s1, s2)))
    ang_sig=sum_aom_r/(m*b)

    angle=np.arctan(np.imag(ang_sig)/np.real(ang_sig))

    # print("\nAngle of Mean - Reuse method for CFO Est. Done. \n")

    return  offset_to_hz(angle/(psi*q))

def cfo_est_aom_nr(signal, b, q, m, psi): #angle of mean with no reuse 

    sum_aom_nr=0
    k=np.arange(1,p+1)
    for i in range(1, b+1):
        a1=2*q*i-2*q+k
        a2=2*q*i-q+k
        a4=np.where(a1<len(signal),a1,a1-len(signal))
        a5=np.where(a2<len(signal),a2,a2-len(signal)) 
        a6=np.where(a4<len(signal),a4,len(signal)-1)
        a7=np.where(a5<len(signal),a5,len(signal)-1)
        s1=np.take(signal, a6)
        s2=np.take(signal, a7) 
        sum_aom_nr+=(np.sum(autocorrelate_phi(s1,s2)))
    ang_sig=sum_aom_nr/(m*b)

    angle=np.arctan(np.imag(ang_sig)/np.real(ang_sig))

    # print("\nAngle of Mean - NonReuse method for CFO Est. Done. \n")

    return  offset_to_hz(angle/(psi*q))

def cfo_est_moa_r(signal, p, b_dash, m, q, psi, b): #mean of angle with reuse 

    angle_sum=0
    k=np.arange(1,p+1)
    for i in range(1, b_dash+1):
        a1=q*i-q+k
        a2=q*i+k
        a4=np.where(a1<len(signal),a1,a1-len(signal))
        a5=np.where(a2<len(signal),a2,a2-len(signal))
        a6=np.where(a4<len(signal),a4,len(signal)-1)
        a7=np.where(a5<len(signal),a5,len(signal)-1)
        s1=np.take(signal, a6)
        s2=np.take(signal, a7) 
        sum_moa_r=(np.sum(autocorrelate_phi(s1,s2)))/m
        angle=np.arctan(np.imag(sum_moa_r)/np.real(sum_moa_r))
        angle_sum+=angle 

    # print("\nMean of Angle - Reuse method for CFO Est. Done. \n")

    return  offset_to_hz(angle_sum/(psi*q*b))

def cfo_est_moa_nr(signal, p, q, b,psi, m): #mean of angle without reuse 

    angle_sum=0
    k=np.arange(1,p+1)
    for i in range(1, b+1):
        a1=2*q*i-2*q+k
        a2=2*q*i-q+k
        a4=np.where(a1<len(signal),a1,a1-len(signal))
        a5=np.where(a2<len(signal),a2,a2-len(signal))
        a6=np.where(a4<len(signal),a4,len(signal)-1)
        a7=np.where(a5<len(signal),a5,len(signal)-1)
        s1=np.take(signal, a6)
        s2=np.take(signal, a7)
        sum_moa_nr=(np.sum(autocorrelate_phi(s1,s2)))/m
        angle=np.arctan(np.imag(sum_moa_nr)/np.real(sum_moa_nr))
        angle_sum+=angle 

    # print("\nMean of Angle - NonReuse method for CFO Est. Done. \n")

    return  offset_to_hz(angle_sum/(psi*q*b))

def cfo_est_conventional(signal, p, q, m, psi, d): #typical method for cfo 

    l_dash=int((len(signal)-1)/d)
    sum_phi=0
    i=np.arange(0,p-1)
    a1=q*i+l_dash
    a2=q*i+l_dash+q
    a4=np.where(a1<len(signal),a1,a1-len(signal))
    a5=np.where(a2<len(signal),a2,a2-len(signal))
    s1=np.take(signal, a4)
    s2=np.take(signal, a5)
    sum_phi=np.sum(autocorrelate_phi(s1,s2))

    # print("\nConventional method for CFO Est. Done. \n")

    return  offset_to_hz((np.arctan((np.imag(sum_phi))/(m*(np.real(sum_phi)))))/(q*psi))

def cfo_est_mle(signal, d, b, q, psi): #MLE method

    l_dash=int((len(signal)-1)/d)
    w_mle=0
    for mu in range(1, b+1):
        sum_mle=0
        p=np.arange(mu, b+1)
        a1=l_dash+q*(p-mu)
        a2=l_dash+q*(p-mu)+mu*q 
        a4=np.where(a1<len(signal),a1,a1-len(signal))
        a5=np.where(a2<len(signal),a2,a2-len(signal))
        a6=np.where(a4<len(signal),a4,len(signal)-1)
        a7=np.where(a5<len(signal),a5,len(signal)-1)
        s1=np.take(signal, a6)
        s2=np.take(signal, a7)
        sum_mle=np.sum(autocorrelate_phi(s1,s2))
        angle=np.arctan(np.imag(sum_mle)/np.real(sum_mle))
        w_mle+=(angle/mu)

    # print("\nMLE method for CFO Est. Done. \n")

    return  offset_to_hz(w_mle/(q*psi))
