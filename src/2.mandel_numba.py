import numpy as np
from PIL import Image
import time
from numba import jit


@jit
def mandelbrot(c, maxiter):
    z = c
    for n in range(maxiter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return 0


@jit
def mandelbrot(creal, cimag, maxiter):
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
            # pv = mandelbrot(r1[i] + 1j * r2[j], maxiter)
            pv = mandelbrot(r1[i], r2[j], maxiter)
            img.putpixel((i, j), (pv << 21) + (pv << 10) + pv * 8)


bitmap = Image.new("RGB", (1000, 1000), "white")

start = time.time()
mandelbrot_set(-2.0, 0.5, -1.25, 1.25, bitmap, 100)
print("执行时间 {} 秒".format(round(time.time() - start, 2)))

# start = time.time()
# mandelbrot_set(-0.74877, -0.74872, 0.06505, 0.06510, bitmap, 2000)
# print("执行时间 {} 秒".format(round(time.time() - start, 2)))

bitmap.show()
