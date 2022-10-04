import time
from src.utils import *
from numba import jit, guvectorize, complex128, int32
import math


def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, maxiter):
    re = np.linspace(xmin, xmax, width, dtype=np.float64)
    im = np.linspace(ymin, ymax, height, dtype=np.float64)
    c = re + im[:, None]*1j

    n3 = mandelbrot_numpy(c, maxiter)

    # To handle row exchange issue.
    rows, row = n3.shape[0], math.floor(n3.shape[0]/2)
    for i in range(row):
        n3[[i, rows - 1 - i], :] = n3[[rows - 1 - i, i], :]

    return n3


@jit(int32(complex128, int32))
def mandelbrot(c, maxiter):
    real = 0
    imag = 0
    for n in range(maxiter):
        nreal = real * real - imag * imag + c.real
        imag = 2 * real * imag + c.imag
        real = nreal
        if real * real + imag * imag > 4.0:
            return n
    return 0


@guvectorize([(complex128[:], int32[:], int32[:])], '(n),()->(n)', target='parallel')
def mandelbrot_numpy(c, maxit, output):
    maxiter = maxit[0]
    for i in range(c.shape[0]):
        output[i] = mandelbrot(c[i], maxiter)


width = 1000
height = 1000
max_iter = 100

start = time.time()
n = mandelbrot_set(-2.0, 0.5, -1.25, 1.25, width, height, max_iter)
img = get_image(n, create_palette())
print("迭代执行时间 {} 秒".format(round(time.time() - start, 2)))

img.show()




