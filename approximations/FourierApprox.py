from scipy import fft
import numpy as np


def fourier1d_approx(func, sigma, kern):
    """
    Computes approximation of gaussian filter using Fast Fourier Transform
    :param func: (1d np.array)
        initial function with noise
    :param sigma: (int)
        standard deviation for gaussian distribution.
        Used only for unification with other approximation functions
    :param kern: (1d np.array)
        preprocessed Gaussian Kernel
    :return:
        1d convoluted np.array using Fourier method
    """

    # get shape of intermediate array
    kern_len = kern.shape[0]
    func_len = func.shape[0]
    s = kern_len + func_len - 1
    # perform Discrete Fourier Transform on function and gaussian kernel
    f_func = fft.rfft(func, s)
    f_kern = fft.rfft(kern, s)
    # multiply in Frequency domain and perform Inverse Fourier Transform
    smoothed_func = fft.irfft(np.multiply(f_func, f_kern))
    # deleting padded elements
    smoothed_func = smoothed_func[kern_len//2: (kern_len + 1)//2 + func_len - 1]
    return smoothed_func
