import numpy as np

def poly1d_approx(func, sigma, gauss_kernel):
    """
   Computes approximation of gaussian filter using polynomial method
    :param func: (1d np.array)
        initial function with noise
    :param sigma: (int)
        standard deviation for gaussian distribution
    :param kern: (1d np.array)
        preprocessed Gaussian Kernel
    :return:
        1d convoluted np.array using Polynomial Method
    """

    k = len(gauss_kernel) // 2

    # манипуляции с ядром
    x_g = range(-k, k + 1)

    x_1 = [i for i in x_g if (i < -1 * sigma)]
    x_2 = [i for i in x_g if (i >= -1 * sigma and i <= 1 * sigma)]
    x_3 = [i for i in x_g if (i > 1 * sigma)]

    g_1 = [gauss_kernel[i] for i in range(0, len(x_1))]
    g_2 = [gauss_kernel[i] for i in range(len(x_1), len(x_1) + len(x_2))]
    g_3 = [gauss_kernel[i] for i in range(len(x_1) + len(x_2), len(x_1) + len(x_2) + len(x_3))]

    aprx_1 = np.polyfit(x_1, g_1, 2)  # [w_2, w_1, w_0]
    aprx_2 = np.polyfit(x_2, g_2, 2)
    aprx_3 = np.polyfit(x_3, g_3, 2)

    conv_final = [0 for i in range(0, len(func))]
    I_add = [0] * k + list(func) + [0] * k
    for aprx, h in list(zip([aprx_1, aprx_2, aprx_3], range(1, 4))):
        w_2, w_1, w_0 = aprx
        if h == 1:
            k1 = 0
            k2 = len(g_1)
            shift1 = 0
            shift2 = -(k + k - (k2 - k1))
        if h == 2:
            k1 = len(g_1)
            k2 = len(g_1) + len(g_2)
            shift1 = k1
            shift2 = -k1 + 1
        if h == 3:
            k1 = len(g_1) + len(g_2)
            k2 = 2 * k + 1
            shift1 = k1
            shift2 = 1

        j = k
        # 0 степень
        conv_0 = []
        sums_0 = []
        # первый элемент свертки
        c_last = sum(I_add[k1:k2])
        conv_0.append(c_last)
        for i in range(j, len(I_add) - k - 1):
            sum_0 = c_last - I_add[i - k + shift1]
            c = sum_0 + I_add[i + k + shift2]
            sums_0.append(sum_0)
            conv_0.append(c)
            c_last = c

        conv_0 = [w_0 * i for i in conv_0]

        # первая степень
        conv_1 = []
        sums_1 = []

        y_1 = [i for i in range(k1 - k, k2 - k)]
        sums_w0 = [i for i in sums_0]

        c_last = sum([y_1[i - k1] * I_add[i] for i in range(k1, k2)])
        conv_1.append(c_last)

        for i in range(j, len(I_add) - k - 1):
            sum_1 = c_last - I_add[i - k + shift1] * y_1[0] - sums_w0[i - k]
            sums_1.append(sum_1)
            c = sum_1 + I_add[i + k + shift2] * y_1[-1]
            conv_1.append(c)
            c_last = c

        conv_1 = [w_1 * i for i in conv_1]
        # вторая степень
        conv_2 = []

        y_2 = [i ** 2 for i in range(k1 - k, k2 - k)]
        sums_w1 = [i for i in sums_1]
        sums_w0 = [i for i in sums_0]

        c_last = sum([y_2[i - k1] * I_add[i] for i in range(k1, k2)])
        conv_2.append(c_last)

        for i in range(k, len(I_add) - k - 1):
            c = c_last - I_add[i - k + shift1] * y_2[0] + I_add[i + k + shift2] * y_2[-1] - 2 * sums_w1[i - k] - \
                sums_w0[i - k]
            conv_2.append(c)
            c_last = c

        conv_2 = [w_2 * i for i in conv_2]
        conv_final = [conv_final[i] + conv_0[i] + conv_1[i] + conv_2[i] for i in range(len(func))]

    return conv_final
