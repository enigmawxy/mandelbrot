import numpy as np
from PIL import Image
import time
from numba import vectorize, complex64, boolean, jit


@vectorize([boolean(complex64)])
def f(z):
    return (z.real * z.real + z.imag * z.imag) < 4.0


@vectorize([complex64(complex64, complex64)])
def g(z, c):
    return z * z + c


@jit
def mandelbrot_numpy(c, maxiter):
    output = np.zeros(c.shape, np.int)
    z = np.empty(c.shape, np.complex64)
    for it in range(maxiter):
        notdone = f(z)
        output[notdone] = it
        z[notdone] = g(z[notdone], c[notdone])
    output[output == maxiter - 1] = 0

    return output


def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:, None]*1j

    n3 = mandelbrot_numpy(c, maxiter)

    return r1, r2, n3.T


def matrix_p(p_v=1):
    pv = int(p_v)
    return (pv << 21) + (pv << 10) + pv * 8


width = 1000
height = 1000

bitmap = Image.new("RGB", (width, height), "white")
start = time.time()
re, rm, n = mandelbrot_set(-2.0, 0.5, -1.25, 1.25, width, height, 100)
print("执行时间 {} 秒".format(round(time.time() - start, 2)))
start = time.time()
nr = np.frompyfunc(matrix_p, 1, 1)(n)
print("执行时间 {} 秒".format(round(time.time() - start, 2)))
start = time.time()
for i in range(width):
    for j in range(height):
        bitmap.putpixel((i, j), nr[i][j])

print("执行时间 {} 秒".format(round(time.time() - start, 2)))
bitmap.show()
