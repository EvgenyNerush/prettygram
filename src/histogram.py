import numpy as np
import h5py

# This function places the bin boundaries such that the number of particles is n_in_bin in every bin,
# (thus N % n particles are dropped from the consideration, where N = len(xs)
# the overall number of particles). The positions of the leftmost and the rightmost
# boundaries are slightly shifted from minimum and maximum of the considered xs values such that
# for all the bins the distribution function f can be roughly computed from the position of the
# left and the rignt bin boundaries b_l and b_r as
#     f = \frac{n_in_bin}{b_r - b_l}.
# The complexity of this histogram function is O(N \log N) (operations).
#
# n_in_bin is the desired number of particles per bin
# xs is the array of coordinates of the particles
#
# momentum_j is the sum of (x - x_0)**j for particles in a bin, with x0 is the bin center;
# For instance, momentum_0 is always equal to n_in_bin.
#
# correction = 0 turns off corrections
# return (positions of the bin borders, n_in_bin)
#
# correction = 2 turns on second order corrections
# return (positions of the bin borders, n_in_bin, momentum_1 in bins, momentum_2 in bins)
#
# correction = 4 turns on corrections up to the fourth order
# return (positions of the bin borders, n_in_bin, momentum_1, momentum_2, momentum_3, momentum_4)
def histogram(n_in_bin, xs, correction = 4):

    l = len(xs)
    bs = [] # boundaries

    if n_in_bin >= 1 and l >= n_in_bin: # the bin width cann't be found otherwise
        ys = np.copy(xs)
        np.random.shuffle(ys)
        ys = ys[0:n_in_bin*(l//n_in_bin)]
        ys.sort()

        # the leftmost boundary
        bs.append(ys[0]);
        into = 0;
        for i in range( len(ys) - 1 ):
            into += 1
            if into == n_in_bin:
                into = 0
                bs.append(0.5 * (ys[i] + ys[i + 1]))

        # the rightmost boundary
        bs.append(ys[-1])

        # correction of the leftmost and the rightmost boundaries
        if len(bs) == 2:
            d = (bs[1] - bs[0]) / (n_in_bin - 1)
            bs[0] -= 0.5 * d
            bs[1] += 0.5 * d
        elif len(bs) > 2:
            d1 = (bs[1] - bs[0]) / (n_in_bin - 0.5);
            d2 = (bs[-1] - bs[-2]) / (n_in_bin - 0.5);
            bs[0]  -= 0.5 * d1;
            bs[-1] += 0.5 * d2;

        bs = np.array(bs)

        if correction < 2:
            return (bs, n_in_bin)

        x_vals = 0.5 * ( bs[1:] + bs[:-1] ) # bin centers
        dx = bs[1:] - bs[:-1]             # bin widths
        hw = dx / 2                       # halfwidths

        momentum_1 = []
        momentum_2 = []
        for i in range(len(x_vals)):
            j = n_in_bin * i
            momentum_1.append( sum( ( ys[j:(j + n_in_bin)] - x_vals[i] )    ) )
            momentum_2.append( sum( ( ys[j:(j + n_in_bin)] - x_vals[i] )**2 ) )
        momentum_1 = np.array(momentum_1)
        momentum_2 = np.array(momentum_2)

        if correction < 4:
            return (bs, n_in_bin, momentum_1, momentum_2)

        momentum_3 = []
        momentum_4 = []
        for i in range(len(x_vals)):
            j = n_in_bin * i
            momentum_3.append( sum( ( ys[j:(j + n_in_bin)] - x_vals[i] )**3 ) )
            momentum_4.append( sum( ( ys[j:(j + n_in_bin)] - x_vals[i] )**4 ) )
        momentum_3 = np.array(momentum_3)
        momentum_4 = np.array(momentum_4)

        return (bs, n_in_bin, momentum_1, momentum_2, momentum_3, momentum_4)
    else:
        print("histogram error: number of particles less than n_in_bin or n_in_bin < 1")

# compute approximation (bin_centers, f, f', f'', f''', f'''') for given values of bin boundaries and momentums
def approximation(hist):
    correction = len(hist) - 2
    bs = hist[0]
    n_in_bin = hist[1]
    x_vals = 0.5 * (bs[1:] + bs[:-1]) # bin centers
    if correction < 2:
        dx = bs[1:] - bs[:-1]
        f_vals = n_in_bin / dx
        return (x_vals, f_vals)
    elif correction < 4:
        momentum_1 = hist[2]
        momentum_2 = hist[3]
        dx = bs[1:] - bs[:-1]
        f_vals = 9/4 * ( n_in_bin / dx - 20/3 * momentum_2 / dx**3 )
        fs_vals = 12 * momentum_1 / dx**3
        fss_vals = 24 / dx**3 * ( n_in_bin - dx * f_vals )
        return (x_vals, f_vals, fs_vals, fss_vals)
    else:
        momentum_1 = hist[2]
        momentum_2 = hist[3]
        momentum_3 = hist[4]
        momentum_4 = hist[5]
        hw = ( bs[1:] - bs[:-1] ) / 2
        a = 225/64
        b = -525/32
        c = 945/64
        f_vals = 1/2 * ( a * n_in_bin / hw + b * momentum_2 / hw**3 + c * momentum_4 / hw**5 )
        fs_vals = 15/8 * ( 5 * momentum_1 / hw**3 - 7 * momentum_3 / hw**5 )
        fss_vals = 15/4 * ( 5 * n_in_bin / hw**3 - 7 * momentum_2 / hw**5 - 16/3 * f_vals / hw**2 )
        fsss_vals = -105/4 * ( 3 * momentum_1 / hw**5 - 5 * momentum_3 / hw**7 )
        fssss_vals = -105 * ( 3 * n_in_bin / hw**5 - 5 * momentum_2 / hw**7 - 8/3 * f_vals / hw**4 )
        return (x_vals, f_vals, fs_vals, fss_vals, fsss_vals, fssss_vals)

# the normalization is the same as in plt.hist: all f values are devided by a number of bins;
# automatically sends plot to the bottom layer
def plot_approximation(ax, hist, y2 = 0, color = '#ffe6a6', label = None, n_for_bin_shape = 7):
    bs = hist[0]
    n_bins = len(bs) - 1
    appr = approximation(hist)
    correction = len(hist) - 2
    if correction < 2:
        _, fs = appr
        ys = [y / n_bins for f in fs for y in [f, f]]
        xs = [x for b in bs[1:-1] for x in [b, b]]
        xs.insert(0, bs[0])
        xs.append(bs[-1])
        #ax.fill_between(xs, ys, y2 = y2, color = color, label = label)
        return (xs, ys)
    elif correction < 4:
        x0, f, fs, fss = appr
        bs_pairs = zip(bs[:-1], bs[1:])
        local_xs = [ np.linspace(b1, b2, n_for_bin_shape) for b1, b2 in bs_pairs ]
        a = zip(local_xs, x0, f, fs, fss)
        local_approximations = [ f + fs * (x - x0) + fss * (x - x0)**2 / 2 for local_xs, x0, f, fs, fss in a for x in local_xs ]
        xs = [x for lx in local_xs for x in lx]
        ys = [y / n_bins for y in local_approximations]
        #ax.fill_between(xs, ys, y2 = y2, color = color, label = label)
        return (xs, ys)
    else:
        x0, f, fs, fss, fsss, fssss = appr
        bs_pairs = zip(bs[:-1], bs[1:])
        local_xs = [ np.linspace(b1, b2, n_for_bin_shape) for b1, b2 in bs_pairs ]
        a = zip(local_xs, x0, f, fs, fss, fsss, fssss)
        local_approximations = [ f + fs * (x - x0) + fss * (x - x0)**2 / 2 + fsss * (x - x0)**3 / 6 + fssss * (x - x0)**4 / 24 for local_xs, x0, f, fs, fss, fsss, fssss in a for x in local_xs ]
        xs = [x for lx in local_xs for x in lx]
        ys = [y / n_bins for y in local_approximations]
        #ax.fill_between(xs, ys, y2 = y2, color = color, label = label)
        return (xs, ys)

## returns histogram values as histogram function above, for correction = 4
#def read_spectrum(filename):
#    f = h5py.File(filename, 'r')
#    dset = f['spectrum']
#    extent = dset.attrs.get('extent', '')
#    data = dset[:]
#    n_in_bin = extent[0]
#    f.close()
#
#    bs = data[0][:]
#    momentum_1 = data[1][:-1]
#    momentum_2 = data[2][:-1]
#    momentum_3 = data[3][:-1]
#    momentum_4 = data[4][:-1]
#
#    return (bs, n_in_bin, momentum_1, momentum_2, momentum_3, momentum_4)

def read_spectrum(filename):
    f = h5py.File(filename, 'r')
    dset = f['spectrum']
    data = dset[:]
    f.close()

    xs = data[0][:]
    ys = data[1][:]

    return (xs, ys)

