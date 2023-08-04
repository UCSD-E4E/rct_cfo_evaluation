'''
BLUE CFO Estimator
'''
import numpy as np

def autocorrelate_phi(signal: np.ndarray, signal_shift: np.ndarray):
    '''
    Helper function to compute autocorrelation of a signal

    Inputs:
        signal: signal to autocorrelate
        signal_shift: shift to apply to signal
    Outputs:
        autocorrelated signal
    '''
    return signal * (np.real(signal_shift) - np.imag(signal_shift))

def cfo_est_blue(signal: np.ndarray, f_s: int = 1000000, repetitions: int = 32):
    '''
    Estimate CFO using BLUE

    Inputs:
        signal: ndarry containing signal data
        f_s: sampling frequency
        repetitions: repetitions of signal data in pilot block
    Outputs:
        CFO Estimate, in cycles per second
    '''

    total_n = 500 #len(signal)
    samples_per_sec = int(total_n/repetitions)
    k = int(repetitions/2)
    fsum = 0
    last_phiu = -1
    phiu = -1

    for i in range( int(repetitions/2) ):
        mu_1 = np.arange(i * samples_per_sec, total_n)
        mu_2 = mu_1 - i * samples_per_sec
        signal_1 = np.take( signal, np.where(mu_1 < len(signal), mu_1, mu_1 - len(signal)) )
        signal_2 = np.take( signal, np.where(mu_2 < len(signal), mu_2, mu_2 - len(signal)) )
        phi_bu = np.sum(autocorrelate_phi(signal_1, signal_2)) / (total_n - i * samples_per_sec)

        if i >= 1:
            current = np.arctan(np.imag(phi_bu) / np.real(phi_bu))
            previous = np.arctan(np.imag(last_phiu) / np.real(last_phiu))
            phiu = (current - previous) % (2 * np.pi)
        else:
            phiu = phi_bu
        last_phiu = phiu

        w_u = ( ((3*(repetitions - i) * (repetitions - i + 1) - (k * (repetitions - k)))) /
                (k * (4*np.square(k) - 6*repetitions*k + 3*np.square(repetitions) - 1)) )
        fsum += (w_u * phiu)

    return fsum * repetitions * f_s / (2 * np.pi)
