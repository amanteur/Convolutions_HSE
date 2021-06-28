def discrete1d_approx(func, sigma, kernel):
    """
    Computes approximation of gaussian filter using Discrete Method
    :param func: (1d np.array)
        initial function with noise
    :param sigma: (int)
        standard deviation for gaussian distribution.
        Used only for unification with other approximation functions
    :param kern: (1d np.array)
        preprocessed Gaussian Kernel
    :return:
        1d convoluted np.array using Discrete Method
    """

    # initialize a convolution array
    conv = []

    # loop for computing convolution array
    for i in range(len(func) + len(kernel) - 1):
        summa = 0
        for k in range(len(func)):
            b_ind = i - k
            if 0 <= b_ind < len(kernel):
                summa += func[k] * kernel[b_ind]
        conv.append(summa)
    # deleting padded elements
    return conv[len(kernel) // 2 + 1:(len(kernel) + 1) // 2 + len(func)]
