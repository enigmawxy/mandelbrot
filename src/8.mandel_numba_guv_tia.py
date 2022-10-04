import time
from src.utils import *
from numba import jit, guvectorize, complex128, int32, float32
import math


def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, maxiter):
    if width > height:
        scale = width / height
    else:
        scale = 1

    re = scale * np.linspace(xmin, xmax, width, dtype=np.float64)
    im = np.linspace(ymin, ymax, height, dtype=np.float64)
    c = re + im[:, None]*1j

    # np.seterr(invalid='ignore')
    n3 = mandelbrot_numpy(c, maxiter)

    # To handle row exchange issue.
    rows, row = n3.shape[0], math.floor(n3.shape[0]/2)
    for i in range(row):
        n3[[i, rows - 1 - i], :] = n3[[rows - 1 - i, i], :]

    return n3


@jit(float32(complex128, int32, int32))
def mandelbrot(c, maxiter, bailout):
    sum, sum2 = 0, 0
    bailout_squared = bailout * bailout
    mandelbrotPower = 2

    ac = np.sqrt(c.real * c.real + c.imag * c.imag)
    il = 1.0 / np.log(mandelbrotPower)
    lp = np.log(np.log(bailout) / mandelbrotPower)
    real, imag = 0, 0

    for n in range(maxiter):
        x = real * real - imag * imag + c.real
        y = 2 * real * imag + c.imag
        magnitude = x * x + y * y
        if magnitude > bailout_squared:
            break

        real, imag = x, y
        sum2 = sum
        # if n is not 0 and n is not (maxiter - 1):
        if n != 0 and n != (maxiter - 1):
            tr = real - c.real
            ti = imag - c.imag
            az2 = np.sqrt(tr * tr + ti * ti)
            lowbound = abs(az2 - ac)
            sum += ((np.sqrt(real * real + imag * imag) - lowbound) / (az2 + ac - lowbound))
            # print(n, sum)

    if n == maxiter - 1:
        # inside
        return 0.
    else:
        if n == 0:
            sum = np.inf
        else:
            sum = sum / n
        if n == 1:
            sum2 = np.inf
        else:
            sum2 = sum2 / (n - 1)

        log1 = np.log(np.sqrt(real * real + imag * imag))
        f = il * lp - il * np.log(log1)
        index = sum2 + (sum - sum2) * (f + 1.0)
        realiters = index * 255
        if np.isnan(realiters) or np.isinf(realiters):
            return 0.

        return realiters


@guvectorize([(complex128[:], int32[:], float32[:])], '(n),()->(n)', target='parallel')
def mandelbrot_numpy(c, maxit, output):
    maxiter = maxit[0]
    for i in range(c.shape[0]):
        output[i] = mandelbrot(c[i], maxiter, 100)


width = 1024
# height = 768
max_iter = 10000
xmin, xmax = (-2.0, 0.5)
ymin, ymax = (-1.25, 1.25)
# xmin, xmax = (-0.7, 0.3)
# ymin, ymax = (-0.3, 1.3)

# xmin, xmax = (-0.8596296296296296, -0.5396296296296296)
# ymin, ymax = (0.2422222222222222, 0.4422222222222222)

height = np.int(np.abs(ymax - ymin) * width / (xmax - xmin))

print(width, height)

start = time.time()
np.random.seed(10)
palette = create_palette()
n = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)

img = render_image(n, palette)
print("迭代执行时间 {} 秒".format(round(time.time() - start, 2)))
img.save("pics/guv_tia_{}.png".format(time.strftime("%Y-%m-%d-%H:%M:%S")), "PNG", optimize=True)
img.show()


