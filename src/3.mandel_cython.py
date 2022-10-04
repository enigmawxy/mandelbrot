import mandel_cython as mc
from PIL import Image
import time

ImageHeight = 1080
ImageWidth = 768
min_re = -2.0
max_re = 0.2
min_im = -1.5
# min_re = -0.74877
# max_re = -0.74872
# min_im = 0.06505
max_im = min_im + (max_re-min_re) * ImageHeight / ImageWidth
scale = 1.6
re_factor = scale * (max_re-min_re)/(ImageWidth-1)
im_factor = scale * (max_im-min_im)/(ImageHeight-1)
max_iterations = 300


def get_complex_map(x, y):
    return min_re + x * re_factor, max_im-y*im_factor


def mandelbrot(image, max_iter=10):
    w = image.size[0]
    h = image.size[1]
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
            if int(n) is (max_iter-1):
                image.putpixel((x, y), (0 % 4 * 64, 0 % 8 * 32, 0 % 16 * 16))
            else:
                # print(x, y, get_complex_map(x, y))
                image.putpixel((x, y), (n % 4 * 64, n % 8 * 32, n % 16 * 16))


img = Image.new("RGB", (ImageHeight, ImageWidth), "white")
start = time.time()
mc.get_mandelbrot_set0(min_re, re_factor, max_im, im_factor, img, max_iterations)
# mc.get_mandelbrot_set(min_re, re_factor, max_im, im_factor, img, max_iterations)
# mandelbrot(img, max_iterations)
current = round(time.time() - start, 2)
print("执行时间 {} 秒".format(current))
# img.save("{}_{}_{}.png".format(time.strftime("%Y-%m-%d-%H:%M:%S"), current, max_iterations), "PNG", optimize=True)
img.show()
