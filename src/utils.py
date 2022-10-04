import numpy as np
import random
import math
from PIL import Image


def matrix_p(p_v=1):
    pv = int(p_v)
    return (pv << 21) + (pv << 10) + pv * 8


def render_color(n):
    return np.frompyfunc(matrix_p, 1, 1)(n)


def get_color(palette):
    # palette = create_palette()

    def color(i):
        return palette[i % 256]

    return color


def clamp(x):
    return max(0, min(x, 255))


def create_palette():
    palette = [(0, 0, 0)]
    red_b = 2 * math.pi / (random.randint(0, 128) + 128)
    red_c = 256 * random.random()
    green_b = 2 * math.pi / (random.randint(0, 128) + 128)
    green_c = 256 * random.random()
    blue_b = 2 * math.pi / (random.randint(0, 128) + 128)
    blue_c = 256 * random.random()

    # print(red_b, red_c)
    # print(green_b, green_c)
    # print(blue_b, blue_c)

    for i in range(256):
        r = clamp(int(256 * (0.5 * math.sin(red_b * i + red_c) + 0.5)))
        g = clamp(int(256 * (0.5 * math.sin(green_b * i + green_c) + 0.5)))
        b = clamp(int(256 * (0.5 * math.sin(blue_b * i + blue_c) + 0.5)))
        palette.append((r, g, b))

    return palette


def get_image(n, palette=None):
    if palette is None:
        r, g, b = np.frompyfunc(get_color(create_palette()), 1, 3)(n)
    else:
        r, g, b = np.frompyfunc(get_color(palette), 1, 3)(n)
    img_array = np.dstack((r, g, b))
    return Image.fromarray(np.uint8(img_array * 255), mode='RGB')


def render_image(n, palette=None):
    if palette is None:
        r, g, b = np.frompyfunc(get_color_by_real(create_palette()), 1, 3)(n)
    else:
        r, g, b = np.frompyfunc(get_color_by_real(palette), 1, 3)(n)
    img_array = np.dstack((r, g, b))
    return Image.fromarray(np.uint8(img_array), mode='RGB')


def get_color_by_real(palette):

    def color(realiters):
        colval1 = int(realiters % 255)
        colval2 = int(colval1 + 1 % 255)
        tweenval = np.modf(realiters)[0]
        if colval1 < 0:
            colval1 = colval1 + 255
        if colval2 < 0:
            colval2 = colval2 + 255

        rv1, gv1, bv1 = get_color(palette)(colval1)
        rv2, gv2, bv2 = get_color(palette)(colval2)
        return int(rv1 + (rv2 - rv1) * tweenval), int(gv1 + (gv2 - gv1) * tweenval), int(bv1 + (bv2 - bv1) * tweenval)

    return color
