import cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_mandelbrot_set0(double min_re, double re_factor, double max_im, double im_factor, image, max_iter=10):
    cdef:
        int w = image.size[0]
        int h = image.size[1]
        int x, y, n
        double c_im,c_re

    for y in range(h):
        c_im = max_im - y * im_factor
        for x in range(w):
            c_re = min_re + x * re_factor
            n = mandelbrot(c_re, c_im, max_iter)
            image.putpixel((x, y), (n % 4 * 64, n % 8 * 32, n % 16 * 16))

cdef int mandelbrot(double creal, double cimag, int maxiter):
    cdef:
        double real2, imag2
        double real = creal, imag = cimag
        int n

    for n in range(maxiter):
        real2 = real*real
        imag2 = imag*imag
        if real2 + imag2 > 4.0:
            return n
        imag = 2* real*imag + cimag
        real = real2 - imag2 + creal

    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot_set(double xmin, double xmax, double ymin, double ymax, int width, int height, int maxiter, pix):
    cdef:
        double[:] r1 = np.linspace(xmin, xmax, width)
        double[:] r2 = np.linspace(ymin, ymax, height)
        int i,j,pv

    for i in range(width):
        for j in range(height):
            pv = mandelbrot(r1[i], r2[j], maxiter)
            pix[i, j] = (pv << 21) + (pv << 10) + pv * 8


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_mandelbrot_set(double min_re, double re_factor, double max_im, double im_factor, image, max_iter=10):
    cdef:
        int w = image.size[0]
        int h = image.size[1]
        int x, y, n
        double c_im,c_re,z_re,z_im,z_re2,z_im2

    for y in range(h):
        c_im = max_im - y * im_factor
        for x in range(w):
            c_re = min_re + x * re_factor
            z_re, z_im = c_re, c_im
            for n in range(max_iter):
                z_re2 = z_re * z_re
                z_im2 = z_im * z_im

                if z_re2 + z_im2 > 4.0:
                    break
                z_im = 2 * z_re * z_im + c_im
                z_re = z_re2 - z_im2 + c_re

            image.putpixel((x, y), (n % 4 * 64, n % 8 * 32, n % 16 * 16))

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef mandelbrot_set1(double xa, double xb, double ya, double yb, int maxiter, image):
    cdef:
        int y, x, i, width, height
        double zy, zx
        complex z, c

    width = image.size[0]
    height = image.size[1]
    for y in range(height):
        zy = y * (yb - ya) / (height - 1) + ya
        for x in range(width):
            zx = x * (xb - xa) / (width - 1) + xa
            i = mandelbrot(zx, zy, maxiter)
            image.putpixel((x, y), (i % 4 * 64, i % 8 * 32, i % 16 * 16))