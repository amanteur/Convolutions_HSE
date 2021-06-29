from main import *


def check_time_memory_diff_n(approx_func, sigmas, n=1000):

    x_1 = (500, np.linspace(0, 1, 500))
    x_2 = (250, np.linspace(0, 1, 250))
    time_list_x = []
    mem_list_x = []
    for x_tuple in [x_1, x_2]:
        time_list = []
        mem_list = []
        x = x_tuple[1]
        np.random.seed(42)
        func = 0.4 * np.sin(5 * np.pi * x)
        # function with some noise
        func_noise = func + np.random.normal(0, 0.4, size=x.shape)
        for sigma in sigmas:
            # compute kernel
            kernel = gauss(sigma)
            # compute eval time
            eval_time = timeit.timeit(lambda: approx_func(func_noise, sigma, kernel), number=n)
            # compute memory usage
            tracemalloc.start()
            approx_func(func, sigma, kernel)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            # appending values to lists
            time_list.append(eval_time)
            mem_list.append(peak / 10 ** 6)
        time_list_x.append((x_tuple[0], time_list))
        mem_list_x.append((x_tuple[0], mem_list))

    return time_list_x, mem_list_x


def plot_different_time_mem(sigmas, stats_dict):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    i = 0
    for stat, funcs in stats_dict.items():
        for func_name, func_n in funcs.items():
            for n_f in func_n:
                axes[i].plot(sigmas, n_f[1], label=func_name + ' n=' + str(n_f[0]))
        i += 1
    axes[0].set_title('Time evaluation with different sigmas')
    axes[0].set_ylabel('Time evaluated')

    axes[1].set_title('Memory usage with different sigmas')
    axes[1].set_ylabel('Peak memory usage, MB')

    handles, labels = axes[1].get_legend_handles_labels()
    lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,0))

    [(ax.set_xlabel('Sigmas'),
      ax.grid(),
      ax.set_yscale('log')) for ax in axes]
    fig.savefig('samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    # initializing different sigmas
    sigmas = [10, 12, 16, 20, 24, 30, 36, 40, 50, 60, 70, 80]
    # different number of repeats for time eval
    rep = [1000, 500, 500, 50]
    approx_dict = {'Fourier': Fourier.fourier1d_approx,
                   'Deriche': Deriche.deriche1d_approx,
                   'Polynomial': Poly.poly1d_approx,
                   'Discrete': Discrete.discrete1d_approx}

    stats_dict = {}
    for (name, approx_func), n in zip(approx_dict.items(), rep):
        stats_dict.setdefault('time_n', {})
        stats_dict.setdefault('mem_n', {})
        stats_dict['time_n'][name], stats_dict['mem_n'][name] = check_time_memory_diff_n(approx_func, sigmas, n)
    stats_dict['time_n']['SciPy'], stats_dict['mem_n']['SciPy'] = check_time_memory_diff_n(test_scipy_gauss, sigmas)
    # plotting time and memory with different n
    plot_different_time_mem(sigmas, stats_dict)
