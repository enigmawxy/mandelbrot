"""
https://en.wikibooks.org/wiki/Fractals/Iterations_in_the_complex_plane/triangle_ineq#cite_note-6
"""
import time
from numba import jit
from utils import *


def mandelbrot(creal, cimag, maxiter, palette, bailout=2):
    sum, sum2 = 0, 0
    mandelbrotPower = 2
    bailout_squared = bailout * bailout

    try:
        ac = np.sqrt(creal * creal + cimag * cimag)
        il = 1.0 / np.log(mandelbrotPower)
        lp = np.log(np.log(bailout) / mandelbrotPower)

        real, imag = 0, 0
        for n in range(maxiter):

            x = real * real - imag * imag + creal
            y = 2 * real * imag + cimag

            magnitude = x * x + y * y
            if magnitude > bailout_squared:
                break
            real = x
            imag = y

            sum2 = sum
            # if n is not 0 and n is not (maxiter - 1):
            if n != 0 and n != (maxiter - 1):
                tr = real - creal
                ti = imag - cimag
                az2 = np.sqrt(tr * tr + ti * ti)
                lowbound = abs(az2 - ac)
                sum += ((np.sqrt(real * real + imag * imag) - lowbound) / (az2 + ac - lowbound))
                # print(n, sum)
        print(n)
        if n == maxiter - 1:
            return 0, 0, 0
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
                return 0, 0, 0
            # print(realiters)
            colval1 = int(realiters % 255)
            colval2 = int(colval1 + 1 % 255)
            tweenval = np.modf(realiters)[0]
            if colval1 < 0:
                colval1 = colval1 + 255
            if colval2 < 0:
                colval2 = colval2 + 255

            rv1, gv1, bv1 = get_color(palette)(colval1)
            rv2, gv2, bv2 = get_color(palette)(colval2)
            rv, gv, bv = (rv1 + (rv2 - rv1) * tweenval), (gv1 + (gv2 - gv1) * tweenval), (bv1 + (bv2 - bv1) * tweenval)

            return int(rv), int(gv), int(bv)

    except Exception as e:
        print('mandelbrot:', e)


@jit
def mandelbrot_set(xmin, xmax, ymin, ymax, bitmap, maxiter):
    # np.random.seed(10)
    width, height = bitmap.size[0], bitmap.size[1]
    img = bitmap.load()
    scale = width / height
    r1 = scale * np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    palette = create_palette()

    np.seterr(invalid='ignore')
    for i in range(width):
        for j in range(height):
            r, g, b = mandelbrot(r1[i], r2[j], maxiter, palette, bailout=2)
            img[i, j] = (r, g, b)

# iterations = round(50 * (math.log(1024 / abs(-0.8596296296296296+0.5396296296296296), 10) ** 1.25))
# print(iterations)


bitmap = Image.new("RGB", (20, 20), "white")
start = time.time()
mandelbrot_set(-2.0, 0.5, -1.25, 1.25, bitmap, 100)
print("执行时间 {} 秒".format(round(time.time() - start, 2)))
# start = time.time()
# mandelbrot_set(-0.8596296296296296, -0.5396296296296296, 0.2422222222222222, 0.4422222222222222, bitmap, 2000)
# print("执行时间 {} 分".format(round((time.time() - start)/60, 2)))
bitmap.save("pics/{}.png".format(time.strftime("%Y-%m-%d-%H:%M:%S")), "PNG", optimize=True)
bitmap.show()
