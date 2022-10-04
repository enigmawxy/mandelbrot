import numpy as np
from PIL import Image
import time
from numba import jit

BAILOUT = 2
p = 2
m = 10
c = -.75 + 0.0j


# Functions for "Smooth Iteration Count"
def smooth_iter(iters, z):
    # return iters + 1 - log(log(abs(z))) / log(m)
    if abs(z) == 0:
        h = 1e-4
    else:
        h = abs(z)
    return iters + 1 + np.log(np.log(BAILOUT) / np.log(h)) / np.log(p)


# Functions for Triangle Inequality Average method for colouring fractals
# Pre: zs has at least two elements
def t(zn_minus1, zn, const):
    abs_zn_minus1 = abs(zn_minus1 ** p)

    mn = abs(abs_zn_minus1 - abs(const))
    Mn = abs_zn_minus1 + abs(const)
    return (abs(zn) - mn) / (Mn - mn)


# to be implemented later
def avg_sum(zs, i, numelems, const):
    if i - numelems == 0:
        return np.inf
    return (sum(t(zs[n - 2], zs[n - 1], const) for n in range(numelems, i)) /
            (i - numelems))


def lin_inp(zs, d, i, num_elems, const=c):
    last_iters_num = i if i < num_elems else num_elems

    return (d * avg_sum(zs, i, last_iters_num, const) +
            (1 - d) * avg_sum(zs[:-1], i, last_iters_num, const))


@jit
def mandelbrot(c, maxiter):
    z = 0
    zs = np.empty(maxiter, dtype='complex64')
    for n in range(maxiter):
        z = z * z + c
        zs[n] = z
        if abs(z) > 2:
            return n, zs

    return maxiter, zs


@jit
def mandelbrot1(creal, cimag, maxiter):
    real = creal
    imag = cimag
    for n in range(maxiter):
        real1 = real * real
        imag1 = imag * imag
        if real1 + imag1 > 4.0:
            return n

        imag = 2*real*imag + cimag
        real = real1 - imag1 + creal

    return 0


@jit
def mandelbrot_set(xmin, xmax, ymin, ymax, img, maxiter):
    width, height = img.size[0], img.size[1]
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    for i in range(width):
        for j in range(height):
            pv, orbit = mandelbrot(r1[i] + 1j * r2[j], maxiter)

            smooth_count = smooth_iter(pv, orbit[pv-1])
            index = lin_inp(orbit, smooth_count % 1.0,
                            pv, m, c)
            # if np.isnan(index) or np.isinf(index):
            #     index = 0
            # if i % 100 == 0:
            #     print(index)

            # pv = mandelbrot(r1[i], r2[j], maxiter)
            img.putpixel((i, j), (pv << 21) + (pv << 10) + pv * 8)


bitmap = Image.new("RGB", (1000, 1000), "white")

start = time.time()
mandelbrot_set(-2.0, 0.5, -1.25, 1.25, bitmap, 100)
print("执行时间 {} 秒".format(round(time.time() - start, 2)))

# start = time.time()
# mandelbrot_set(-0.74877, -0.74872, 0.06505, 0.06510, bitmap, 2000)
# print("执行时间 {} 秒".format(round(time.time() - start, 2)))

bitmap.show()
