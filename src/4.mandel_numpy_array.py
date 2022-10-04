from PIL import Image
import time
from src.utils import *


def mandelbrot_numpy(c, maxiter):
    output = np.zeros(c.shape)
    z = np.zeros(c.shape, np.complex64)
    for it in range(maxiter):
        notdone = np.less(z.real*z.real + z.imag*z.imag, 4.0)
        output[notdone] = it
        z[notdone] = z[notdone]**2 + c[notdone]
    output[output == maxiter-1] = 0

    return output


def mandelbrot_set(xmin, xmax, ymin, ymax, img, maxiter):
    width, height = img.size[0], img.size[1]
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)

    c = r1 + r2[:, None] * 1j
    n3 = mandelbrot_numpy(c, maxiter)

    return n3.T


w, h = 1000, 1000
bitmap = Image.new("RGB", (w, h), "white")

start = time.time()
n = mandelbrot_set(-2.0, 0.5, -1.25, 1.25, bitmap, 100)
print("迭代执行时间 {} 秒".format(round(time.time() - start, 2)))

nr = render_color(n)

start = time.time()
for i in range(w):
    for j in range(h):
        bitmap.putpixel((i, j), nr[i][j])
print("渲染执行时间 {} 秒".format(round(time.time() - start, 2)))
bitmap.show()
