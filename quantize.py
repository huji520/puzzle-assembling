from imageio import imread
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import math

GRAYSCALE = 1
RGB = 2
RGB2YIQ_MATRIX = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.275, -0.321],
                           [0.212, -0.523, 0.311]])
YIQ2RGB_MATRIX = np.linalg.inv(RGB2YIQ_MATRIX)
BITS = 8  # number of bits for pixel
FACTOR = 255  # (1 << BITS) - 1
BINS = 256  # (1 << BITS)
GRAYSCALE_DIM = 2
RGB_DIM = 3


def read_image(filename, representation):
    """
    Get an image and convert to the given representation.
    :param filename: path
    :param representation: 1 for GRAYSCALE, 2 for RGB
    :return: the converted image
    """
    image = imread(filename).astype(np.float64)
    if representation == RGB:
        return image / FACTOR  # According to the instructions, the input image must be RGB
    elif representation == GRAYSCALE:
        return rgb2gray(image) / FACTOR
    else:
        exit()


def imdisplay(filename, representation):
    """
    Display image to the screen with a given representation
    :param filename: path
    :param representation: 1 for GRAYSCALE, 2 for RGB
    """
    plt.imshow(read_image(filename, representation), cmap='gray')
    plt.show()


def is_grayscale(image):
    """
    Determine if the given image is grayscale or not.
    :param image: np array
    :return: true iff the given picture is grayscale
    """
    return image.ndim == GRAYSCALE_DIM or np.allclose(np.equal(image[:, :, 0], image[:, :, 1]),
                                                      np.equal(image[:, :, 1], image[:, :, 2]))


def rgb2yiq(imRGB):
    """
    convert RGB to YIQ
    :param imRGB: image in RGB format
    :return: image in YIQ format
    """
    return imRGB @ RGB2YIQ_MATRIX.T.copy()


def yiq2rgb(imYIQ):
    """
    YIQ to RGB
    :param imYIQ: image in YIQ format
    :return: image in RGB format
    """
    return imYIQ @ YIQ2RGB_MATRIX.T.copy()


def histogram_helper(img):
    """
    calculate histogram of the input image and also after equalization, and fix the image according to the results.
    :param img: grayscale or rgb, integers values in range [0,255]
    :return: [equalized image, histogram of the original image, histogram of the equalized image]
    """
    hist_orig, bounds = np.histogram(img, bins=BINS, range=(0, 255))
    cum = np.cumsum(hist_orig)  # cumulative histogram
    m = hist_orig.nonzero()[0][0]  # first level that not zero
    lut = np.rint(FACTOR * ((cum - cum[m]) / (cum[-1] - cum[m]))).astype(np.uint8)  # normalization + rounding
    img = lut[img]  # fix the original image
    hist_eq, bounds = np.histogram(img, bins=BINS, range=(0, 255))
    return img / FACTOR, hist_orig, hist_eq


def histogram_equalize(im_orig):
    """
    calculate histogram of the input image and also after equalization, and fix the image according to the results.
    If the input is grey picture - just call to histogram_helper. If the input is RGB, call to histogram_helper just
    with the Y channel (convert from rgb to yiq).
    :param im_orig: original picture, float values in range [0,1]
    :return: [equalized image, histogram of the original image, histogram of the equalized image]
    """
    if is_grayscale(im_orig):
        return list(histogram_helper(np.rint(im_orig * FACTOR).astype(np.uint8)))

    # input image is RGB
    yiq = rgb2yiq(im_orig)
    gray_image = yiq[:, :, 0]  # take only the first value of each pixel (the Y channel)
    img, hist_orig, hist_eq = histogram_helper(np.rint(gray_image * FACTOR).astype(np.uint8))
    yiq[:, :, 0] = img  # replace the values with the correct values (after equalization)
    img = yiq2rgb(yiq)
    return [np.clip(img, 0, 1), hist_orig, hist_eq]


def initialize_z(hist_orig, n_quant):
    """
    calculation the borders which divide the histograms into segments
    :param hist_orig: the histogram of the original image
    :param n_quant: number of segments to divide (number of grey levels)
    :return: np array with initialized z values
    """
    cum = np.cumsum(hist_orig)
    section_length = cum[-1] / n_quant
    z = [0]
    for i in range(1, n_quant):
        # we just need the first value that applied this condition
        to_append = np.where(cum >= i * section_length)[0][0]
        while to_append in z:  # avoiding duplicates
            to_append += 1
        z.append(to_append)
    if z[-1] == FACTOR:  # handle with duplicates
        for j in range(FACTOR):
            if z[j] != j:
                z.insert(j, j)
    else:
        z.append(FACTOR)
    return np.array(z)


def calc_z(q, n_quant):
    """
    calculation z with the formula from lecture
    :param q: the values to which each of the segments intensities will map (array)
    :param n_quant: number of segments to divide
    :return: np array with z values
    """
    z = [0]
    for i in range(1, n_quant):
        z.append(math.ceil((q[i - 1] + q[i]) / 2))
    z.append(FACTOR)
    return np.asarray(z)


def calc_q(hist, q, z, n_quant):
    """
    calculation the values to which each of the segments intensities will map
    :param hist: histogram of the original image
    :param q: the values to which each of the segments intensities will map (array)
    :param z: the borders which divide the histograms into segments (array)
    :param n_quant: number of grey levels
    :return: the updated values to which each of the segments intensities will map (array)
    """
    for i in range(n_quant):
        numerator = np.sum(hist[z[i]:z[i + 1]] * np.arange(z[i], z[i + 1], 1))
        denominator = np.sum(hist[z[i]:z[i + 1]])
        q[i] = np.round(numerator / denominator)
    return q


def calc_error(hist, q, z, n_quant):
    """
    calculation the error with the formula from the lecture
    :param hist: histogram of the original image
    :param q: the last q array to override
    :param z: the borders which divide the histograms into segments (array)
    :param n_quant: number of grey levels
    :return: the error (float)
    """
    error = 0
    for i in range(n_quant):
        hist_i = hist[z[i]:z[i+1]]
        e_i = np.power((np.arange(z[i], z[i+1], 1) - q[i]), 2)
        error += np.sum(hist_i * e_i)

    return error


def quantize_helper(image, n_quant, n_iter):
    """
    Helper function for quantize(), that calculate the error and also calculate the vector (extend_z) that map each
    grey level to the new one.
    :param image: the input image in gray scale
    :param n_quant: number of grey levels.
    :param n_iter: is the maximum number of iterations of the optimization procedure (may converge earlier.)
    :return: [image, error] - where the image is the input image after quantization, and the error is the error of
             the quantization procedure.
    """
    error = np.zeros(n_iter)
    hist_orig, bounds = np.histogram(image, bins=BINS, range=(0, 255))
    z = initialize_z(hist_orig, n_quant)  # Initial something reasonable
    q = np.zeros(n_quant)
    i = 0
    for i in range(n_iter):
        q = calc_q(hist_orig, q, z, n_quant)
        last_z = z.copy()
        z = calc_z(q, n_quant)
        error[i] = calc_error(hist_orig, q, z, n_quant)
        if np.array_equal(z, last_z):
            break  # Convergence

    # plt.plot(error[:i+1])
    # plt.show()

    lut = []
    for j in range(1, len(z)):
        lut.extend((z[j] - z[j - 1]) * [q[j - 1]])
    lut.append(q[-1])
    lut = np.rint(lut).astype(np.uint8)

    return lut[image] / FACTOR, error[:i+1]


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given grayscale or RGB image
    :param im_orig: is the input grayscale or RGB image to be quantized (float64 image with values in [0, 1]).
    :param n_quant: number of grey levels.
    :param n_iter: is the maximum number of iterations of the optimization procedure (may converge earlier.)
    :return: [im_quant, error] where im_quant is the quantized output image, and error is an array with shape n_iter or
             less of the total intensities error for each iteration of the quantization procedure.
    """
    if is_grayscale(im_orig):
        image, error = quantize_helper(np.rint(im_orig * FACTOR).astype(np.uint8), n_quant, n_iter)
        return [image, error]

    else:
        # input image is RGB
        yiq = rgb2yiq(im_orig)
        gray_image = yiq[:, :, 0]  # take only the first value of each pixel (the Y channel)
        gray_image, error = quantize_helper(np.rint(gray_image * FACTOR).astype(np.uint8), n_quant, n_iter)
        yiq[:, :, 0] = gray_image

        return [yiq2rgb(yiq), error]


def quantize_rgb(im_orig, n_quant):
    """
    ***BONUS***
    ***NOTE***
    I admit that I found a similar code online, understood it in depth, and rewrite it in my way.
    **********
    This function performs quantization of a given RGB image in range values of [0,1]
    :param im_orig: given image in range [0,1]
    :param n_quant: number of pixel levels.
    :return: quantized output image
    """
    from sklearn.cluster import KMeans
    w = im_orig.shape[1]
    h = im_orig.shape[0]
    im_orig = np.rint(im_orig*FACTOR).astype(np.uint8)
    im_orig_2d = im_orig.reshape((w * h, RGB_DIM))  # reduce dimension to 2D -> [[r,g,b],...,[r,g,b]]
    k_means = KMeans(n_clusters=n_quant).fit(im_orig_2d)
    labels = np.asarray(k_means.labels_)
    lut = np.asarray(k_means.cluster_centers_)
    img_2d = lut[labels]  # example: a = [1,2,3,2,3,0,0], b = [0.2,0.3,0.4,0.5], b[a] = [0.3,0.4,0.5,0.4,0.5,0.2]
    img_reshape = img_2d.reshape((h, w, RGB_DIM)).astype(np.uint8)  # reshape for 3D picture
    return img_reshape / FACTOR
