import numpy as np
from math import exp

def get_conv(func, consts, direction):
    """
    Computes causal or anticausal sequence for Deriche approximation of Gaussian filter.
    :param func: (1d np.array)
        initial function with noise
    :param consts: (list)
        list of constants used for computing causal and anticausal sequences
    :param direction: (bool)
        if True, compute causal sequence, else if False, compute anticausal sequence
    :return:
        1d np.array sequence in one direction
    """

    # get size of initial function
    imax = func.shape[0]

    # get constants
    d2 = consts[0]
    d1 = consts[1]
    n0 = consts[2]
    n1 = consts[3]
    n2 = consts[4]

    # initialize inputs and outputs
    in1 = in2 = 0
    out1 = out2 = 0

    # initialize direction
    if direction:
        rng = range(imax)
    elif not direction:
        rng = range(imax - 1, -1, -1)

    # initialize output sequence
    y = np.zeros(func.shape)

    # main loop for computing output sequence
    for i in rng:
        in0 = func[i]
        out0 = n2 * in2 + n1 * in1 + n0 * in0 + d1 * out1 + d2 * out2

        in2 = in1
        in1 = in0
        out2 = out1
        out1 = out0
        y[i] = out0

    return y


def deriche1d_approx(func, sigma, gauss_kernel=None):
    """
    Computes approximation of gaussian filter using Deriche Method
    :param func: (1d np.array)
        initial function with noise
    :param sigma: (int)
        standard deviation for gaussian distribution
    :param kern: (1d np.array)
        preprocessed Gaussian Kernel
        Used only for unification with other approximation functions
    :return:
        1d convoluted np.array using Deriche Method
    """

    # R. Deriche.
    # Fast Algorithms for Low Level Vision.
    # IEEE Transactions on pattern Analysis and Machine Intelligence, vol. 12, no. 1, pp. 78-87, 1990

    # defining constants
    scale_alpha = 5 / (2 * np.sqrt(np.pi))
    # alpha = scale_alpha / sigma
    alpha = 1.812 / sigma # best value
    e = exp(-alpha)
    e2 = exp(-2 * alpha)

    k = np.power((1 - e), 2) / (1 + 2 * alpha * e - e2)

    pos = [-e2, 2 * e, k, k * (alpha - 1) * e, 0]
    neg = [-e2, 2 * e, 0, k * (alpha + 1) * e, -k * e2]

    # running right to left across the array
    y1 = get_conv(func, consts=pos, direction=True)
    # running left to right across the array
    y2 = get_conv(func, consts=neg, direction=False)

    # summing up causal and anticausal sequence
    y = y1 + y2

    return y
