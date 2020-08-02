from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scipy

# file directory
filedir = './examples/m51/'

# weights
r_w = 1
g_w = 0.7
b_w = 0.7

with fits.open(filedir + 'r.fit') as hdul:
    hdul.info()
    r = hdul[0].data.astype(np.float32)

with fits.open(filedir + 'g.fit') as hdul:
    hdul.info()
    g = hdul[0].data.astype(np.float32)

with fits.open(filedir + 'b.fit') as hdul:
    hdul.info()
    b = hdul[0].data.astype(np.float32)


def absError(a, b):
    return abs(a - b)


def error(r, g, b):
    errorCount = 0
    x_len = len(r)
    y_len = len(r[0])

    x_search = int(x_len / 2)
    y_search = int(y_len / 2)
    margin = 100

    for x in range(x_search - margin, x_search + margin):
        for y in range(y_search - margin, y_search + margin):
            errorCount += absError(r[x][y], g[x][y]) + absError(g[x]
                                                                [y], b[x][y]) + absError(b[x][y], r[x][y])

    return errorCount


def shift(r, g, b, x, y):
    if x == 0 or y == 0:
        return r, g, b

    if x < 0:
        r_1 = r[:x * 2, :]
        g_1 = g[-x:x, :]
        b_1 = b[-x*2:, :]
    else:
        r_1 = r[x*2:, :]
        g_1 = g[x:-x, :]
        b_1 = b[:-x*2, :]

    if y < 0:
        r_2 = r_1[:, :y * 2]
        g_2 = g_1[:, -y:y]
        b_2 = b_1[:, -y*2:]
    else:
        r_2 = r_1[:, y*2:]
        g_2 = g_1[:, y:-y]
        b_2 = b_1[:, :-y*2]

    return r_2, g_2, b_2


def alignImage(r, g, b):
    minX = 0
    minY = 0
    minError = error(r, g, b)

    for i in range(-10, 10):
        for j in range(-10, 10):
            r_s, g_s, b_s = shift(r, g, b, i, j)
            e = error(r_s, g_s, b_s)
            if(e < minError):
                minError = e
                minX = i
                minY = j

    return shift(r, g, b, minX, minY)


r, g, b = alignImage(r, g, b)

rgb = np.dstack(((r * r_w).astype(np.uint32),
                 (g * g_w).astype(np.uint32), (b * b_w).astype(np.uint32)))

plt.imshow(rgb)

plt.show()
