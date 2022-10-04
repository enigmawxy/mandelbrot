import numpy as np
from PIL import Image
import time

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel

complex_gpu = ElementwiseKernel(
    "pycuda::complex<float> *q, int *output, int maxiter",
    """
    {
        float nreal, real = 0;
        float imag = 0;
        output[i] = 0;
        for(int curiter = 0; curiter < maxiter; curiter++) {
            float real2 = real*real;
            float imag2 = imag*imag;
            nreal = real2 - imag2 + q[i].real();
            imag = 2* real*imag + q[i].imag();
            real = nreal;
            if (real2 + imag2 > 4.0f){
                output[i] = curiter;
                break;
                };
        };
    }
    """,
    "complex5",
    preamble="#include <pycuda-complex.hpp>",)


def mandelbrot_gpu(c, maxiter):
    q_gpu = gpuarray.to_gpu(c.astype(np.complex64))
    iterations_gpu = gpuarray.to_gpu(np.empty(c.shape, dtype=np.int))
    complex_gpu(q_gpu, iterations_gpu, maxiter)

    return iterations_gpu.get()


def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:, None]*1j

    # n3 = mandelbrot(c, maxiter)
    n3 = mandelbrot_gpu(c, maxiter)

    return r1, r2, n3.T


def matrix_p(p_v=1):
    pv = int(p_v)
    return (pv << 21) + (pv << 10) + pv * 8


width = 1080
height = 768

bitmap = Image.new("RGB", (width, height), "white")
start = time.time()
# re, rm, n = mandelbrot_set(-2.0, 0.5, -1.25, 1.25, width, height, 80)
re, rm, n = mandelbrot_set(-0.74877, -0.74872, 0.06505, 0.06510, width, height, 2048)
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
