'''BLUE CFO Estimator
'''
import numpy as np

def autocorrelate_phi(signal, signal_shift):
    return signal * (np.real(signal_shift) - np.imag(signal_shift))

def offset_to_hz(x): 
    return ((10**(-np.abs(x)))/20)*(10**5)

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
    return offset_to_hz(1 / ((fsum * u * f_s) / (2 * np.pi)))