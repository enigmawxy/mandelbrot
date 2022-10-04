import time
from src.utils import *
from numba import jit, guvectorize, complex128, int32
import math


def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, maxiter):
    re = np.linspace(xmin, xmax, width, dtype=np.float64)
    im = np.linspace(ymin, ymax, height, dtype=np.float64)
    c = re + im[:, None]*1j

    n3 = np.empty(c.shape, int)
    maxit = np.ones(c.shape, int) * maxiter

    n3 = mandelbrot_numpy(c, maxit)

    # To handle row exchange issue.
    rows, row = n3.shape[0], math.floor(n3.shape[0]/2)
    for i in range(row):
        n3[[i, rows - 1 - i], :] = n3[[rows - 1 - i, i], :]

    return n3


@guvectorize([(complex128[:], int32[:], int32[:])], '(n),()->(n)', target='cuda')
def mandelbrot_numpy(c, maxit, output):
    maxiter = maxit[0]
    for i in range(c.shape[0]):
        creal = c[i].real
        cimag = c[i].imag
        real = creal
        imag = cimag
        output[i] = 0
        for n in range(maxiter):
            real2 = real * real
            imag2 = imag * imag
            if real2 + imag2 > 4.0:
                output[i] = n
                break
            imag = 2 * real * imag + cimag
            real = real2 - imag2 + creal


width = 1000
height = 1000
max_iter = 100

start = time.time()
n = mandelbrot_set(-2.0, 0.5, -1.25, 1.25, width, height, max_iter)
img = get_image(n, create_palette())
print("迭代执行时间 {} 秒".format(round(time.time() - start, 2)))

img.show()




