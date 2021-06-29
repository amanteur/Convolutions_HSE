import approximations.FourierApprox as Fourier
import approximations.DericheApprox as Deriche
import approximations.DiscreteApprox as Discrete
import approximations.PolynomialApprox as Poly

from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from math import pi, sqrt, exp, log10
import timeit
import tracemalloc


def gauss(sigma=6, truncate=3):
    """
    Creates 1-dimensional Gaussian Kernel.
    :param sigma: (int)
        standard deviation for Gaussian kernel
    :param truncate: (int)
        radius of kernel window
    :return:
      Gaussian 1d Kernel
    """

    # compute range for Gaussian
    r = range(-truncate * sigma, truncate * sigma + 1)

    return np.fromiter((1 / (sigma * sqrt(2 * pi)) * exp(-float(x) ** 2 / (2 * sigma ** 2)) for x in r),
                       dtype=np.float32)


def PSNR(original, compressed):
    """
    Computes Peak Signal-to-Noise Ratio metric
    :param original: (1d np.array)
        initial function without noise
    :param compressed: (1d np.array)
        filtered function with approximation method
    :return: (int)
        Peak Signal-to-Noise Ratio metric
    """

    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    peak = max(original)
    psnr = 20 * log10(peak / sqrt(mse))

    return psnr


def plot_approximation(func, func_noise, approx_func, sigmas):
    """
    Plots approximations via different methods and
    computes errors according to SciPy function and PSNR metric
    :param func: (1d np.array)
        initial function without noise
    :param func_noise: (1d np.array)
        initial function with noise
    :param approx_func: (function)
        function of different methods approximation
    :param sigmas: (list)
        list of sigmas used in computations
    :return: tuple
        Tuple of errors list and PSNR list
    """

    error_list = []
    psnr_list = []
    x = np.linspace(0, 1, func_noise.shape[0])
    n_rows = len(sigmas) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(2 * 5, n_rows * 5))
    for i, sigma in enumerate(sigmas):
        # finding kernel
        kernel = gauss(sigma)
        # acquiring approximation function
        apprx_func = approx_func(func_noise, sigma, kernel)

        # Gauss filter for computing error
        g_func = ndimage.gaussian_filter1d(func_noise, sigma, truncate=3)

        # PSNR metric for evaluating the quality of denoising
        psnr = PSNR(func, apprx_func)
        psnr_list.append(psnr)

        error = np.abs(g_func - apprx_func)
        error_list.append(error.mean())

        # plotting
        if i % 2 == 0:
            axes[i // 2, 0].plot(x, func_noise, 'k', label='Original function', alpha=0.4)
            axes[i // 2, 0].plot(x, apprx_func, '--', label='Approximated', color="purple")
            axes[i // 2, 0].plot(x, g_func, ':', label='Gaussian', color="pink")
            axes[i // 2, 1].plot(x, error, 'k')

            [ax.grid() for ax in axes[i // 2]]
            axes[i // 2, 0].legend()
            axes[i // 2, 0].set_title('Approximation of Gaussian, sigma = {}'.format(sigma))
            axes[i // 2, 1].set_title('Error, mean={:.3f}'.format(error.mean()))

    plt.show()

    return error_list, psnr_list


def test_scipy_gauss(func, sigma, kernel=None):
    """
    Wrap for SciPy gaussian_filter1d used for unification in comparing
    :param func: (1d np.array)
        initial function with noise
    :param sigma: (int)
        standard deviation for gaussian distribution.
        Used only for unification with other approximation functions
    :param kern: (1d np.array)
        preprocessed Gaussian Kernel
    :return:
        SciPy gaussian_filter1d on function
    """

    return ndimage.gaussian_filter1d(func, sigma, truncate=3)


def check_time_memory(func, approx_func, sigmas, n=1000):
    """
    Computes statistics (eval time and memory usage) for approximations with different sigmas
    :param func: (1d np.array)
        initial function with noise
    :param approx_func: (function)
        function of different methods approximation
    :param sigmas: (list)
        list of sigmas used in computations
    :param n: (int)
        maximum number of evaluation the approximation for computing time eval
    :return: (tuple)
        tuple of lists of statistics of approximations
    """

    time_list = []
    mem_list = []
    for sigma in sigmas:
        # compute kernel
        kernel = gauss(sigma)
        # compute eval time
        eval_time = timeit.timeit(lambda: approx_func(func, sigma, kernel), number=n)
        # compute memory usage
        tracemalloc.start()
        approx_func(func, sigma, kernel)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # appending values to lists
        time_list.append(eval_time)
        mem_list.append(peak / 10 ** 6)
    return time_list, mem_list


def plot_stats(sigmas, stats_dict):
    """
    Plots statistics of approximations with different sigmas

    :param sigmas: (list)
        list of sigmas used in computations
    :param stats_dict: (dict)
        dict of stats used in computations
    """

    plt.figure(figsize=(7, 7))
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for (stat, funcs), axe in zip(stats_dict.items(), axes):
        for func_name, func in funcs.items():
            axe.plot(sigmas, func, label=func_name)

    axes[0].set_title('Mean Error of approximation with different sigmas')
    axes[0].set_ylabel('Mean Error')

    axes[1].set_title('Time evaluation with different sigmas')
    axes[1].set_ylabel('Time evaluated')
    axes[1].set_yscale('log')

    axes[2].set_title('Memory usage with different sigmas')
    axes[2].set_ylabel('Peak memory usage, MB')
    axes[2].set_yscale('log')

    axes[3].set_title('Peak Signal-to-Noise Ratio')
    axes[3].set_ylabel('PSNR')

    [(ax.legend(),
      ax.set_xlabel('Sigmas'),
      ax.grid()) for ax in axes]
    plt.show()


if __name__ == '__main__':

    # initial function
    x = np.linspace(0, 1, 500)
    np.random.seed(42)
    func = 0.4 * np.sin(5 * np.pi * x)
    # function with some noise
    func_noise = func + np.random.normal(0, 0.4, size=x.shape)

    # another function for testing
    func_2 = np.zeros(500)
    func_2[250:310] = 1
    # func_noise = func_2

    # plot function and its noised version
    plt.figure(figsize=(5, 5))
    plt.plot(func, 'k')
    plt.plot(func_noise, color='gray', alpha=0.5)
    plt.grid()
    plt.show()

    # initializing different sigmas
    sigmas = [10, 12, 16, 20, 24, 30, 36, 40, 50, 60, 70, 80]
    # different number of repeats for time eval
    rep = [1000, 1000, 1000, 100]
    approx_dict = {'Fourier': Fourier.fourier1d_approx,
                   'Deriche': Deriche.deriche1d_approx,
                   'Polynomial': Poly.poly1d_approx,
                   'Discrete': Discrete.discrete1d_approx}

    #computing all stats
    stats_dict = {}
    for (name, approx_func), n in zip(approx_dict.items(), rep):
        stats_dict.setdefault('error', {})
        stats_dict.setdefault('time', {})
        stats_dict.setdefault('mem', {})
        stats_dict.setdefault('noise', {})
        # plot approximations with different sigmas and compute error and PSNR metrics
        stats_dict['error'][name], stats_dict['noise'][name] = plot_approximation(func, func_noise, approx_func, sigmas)
        # compute time and memory statistics
        stats_dict['time'][name], stats_dict['mem'][name] = check_time_memory(func_noise, approx_func, sigmas, n)
    stats_dict['time']['SciPy'], stats_dict['mem']['SciPy'] = check_time_memory(func_noise, test_scipy_gauss, sigmas)
    # plotting statistics with different sigmas
    plot_stats(sigmas, stats_dict)
