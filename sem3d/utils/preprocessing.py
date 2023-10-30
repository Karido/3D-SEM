# Copyright (c) 2023, Stefan Toeberg.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# (http://opensource.org/licenses/BSD-3-Clause)
#
# __author__ = "Stefan Toeberg, LUH: IMR"
# __description__ = """
#                   reading images and preprocessing of images like
#                   normalization, histogram matching and so on
#                   """


# libraries
import cv2 as cv
import os
import numpy as np
import copy


def get_images(config):
    folder = config["imageFolder"]

    # load images alphabetically sorted
    imageDict = load_images_from_folder(folder)
    imageKeys = sorted(list(imageDict.keys()))
    images = []
    [images.append(imageDict[key]) for key in imageKeys]

    # if desired, perform histogram matching, first image functions as the base image
    if config["histMatch"] == True:
        imagesProcessed = copy.copy(images)
        for i in range(len(images) - 1):
            imagesProcessed[i + 1] = hist_match(
                imagesProcessed[i + 1], imagesProcessed[i]
            )
    else:
        imagesProcessed = images

    # define stereo pair images
    stereoPairs = get_stereo_pairs(imagesProcessed)

    selectedPairs = []
    for i, imgNums in enumerate(config["selectedPairs"]):
        ni1 = imgNums[0]
        ni2 = imgNums[1]
        if config["histMatch"] == True:
            selectedPairs.append(
                [images[ni1], hist_match(images[ni2], images[ni1])])
        else:
            selectedPairs.append([images[ni1], images[ni2]])
        print(".", end="", flush=True)
    print(" {} stereo pair(s) selected.".format(i + 1))

    return stereoPairs, selectedPairs


def load_images_from_folder(folder):
    """
    reads all images as grayscale situated in a folder
    the folder has to contain only image data
    the order of the files should be in the same order as the tilt series
    f.i. starting with -15°, ending with +30°

    folder  - directory as a string, f.i., r'Z:\...\Images'
    --------------------------------------------------
    images  - list that contains images
    """

    images = {}
    i = 0
    print("\n Load Images:")

    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder, filename), cv.IMREAD_GRAYSCALE)

        if img is not None:
            images[filename] = img
            print(".", end="", flush=True)
            i += 1
        else:
            raise Exception("There are no images in the folder.")

    print(" {} images loaded.".format(i))
    return images


def load_data_from_folder(folder):
    """
    reads all npy files situated in a folder
    the folder has to contain only npy files

    folder  - directory as a string, f.i., r'Z:\...\Data'
    --------------------------------------------------
    data    - list that contains numpy data
    """

    data = {}
    print("\n load files:")

    i = 0
    for filename in os.listdir(folder):
        datum = np.load(os.path.join(folder, filename))

        if datum is not None:
            data[filename] = datum
            print(".", end="", flush=True)
            i += 1

    print("\n{} files loaded.".format(i))
    return data


def normalize_intensities(img):
    """
    linear normalization of image intensities according to (I-min)*(255/(max-min))

    img - grayscale uint8 image as numpy array
    --------------------------------------------------
    arr - normalized input array as uint8 numpy array
    """

    arr = np.array(img)
    arr = arr.astype("float")
    minVal = arr.min()
    maxVal = arr.max()
    arr -= minVal
    arr *= 255.0 / (maxVal - minVal)

    return arr.astype("uint8")


def hist_match(source, template):
    """
    matching the histogram of the stereo images may improve the feature
    and dense matching performance,
    function adjusts the pixel values of a grayscale image such that its
    histogram matches the one of a template image

    source   -  np.ndarray
                image to adjust, the histogram is computed
                based on the flattened array
    template -  np.ndarray
                template image;
                can have a different size
    --------------------------------------------------
    matched  -  np.ndarray
                adjusted source image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the unique pixel values and their corresponding indices and
    # counts
    sValues, binIdx, sCounts = np.unique(source, return_inverse=True, return_counts=True
                                         )
    tValues, tCounts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    sQuantiles = np.cumsum(sCounts).astype(np.float64)
    sQuantiles /= sQuantiles[-1]
    tQuantiles = np.cumsum(tCounts).astype(np.float64)
    tQuantiles /= tQuantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interpTvalues = np.interp(sQuantiles, tQuantiles, tValues)

    return np.uint8(interpTvalues[binIdx].reshape(oldshape))


def get_stereo_pairs(images):
    """
    get a list of stereo_pairs for better handling

    images         - list of images in the correct order:
                     f.i. 0°_image, 5°_image, 10° image, ...
    --------------------------------------------------
    stereo_pairs   - list of n stereo_pairs, the list contains a list
                     of two numpy arrays for each stereo pair
    """

    stereoPairs = []

    for i in range(len(images) - 1):
        stereoPairs.append([images[i], images[i + 1]])

    return stereoPairs


def get_templates(I, n):
    """
    returns all possible templates with windowsize (nxn) of a (pxq) numpy
    array in form of a 3-dimensional array

    I           - numpy array of size (p,q), (int64)
    n           - windowsize of template (n,n) (3,5,9,11,... possible), (int)
    --------------------------------------------------
    templates   - numpy array of size ( (p-l)*(q-l), n, n )
                  containing all valid templates
    """

    # width of the ignored edge values
    l = n // 2

    # xyIdx contains the indices of each array position
    xyIdx = np.mgrid[0: I.shape[0], 0: I.shape[1]]
    xIdx = xyIdx[1]
    yIdx = xyIdx[0]

    # mask is used to ignore the edge values
    mask = np.ones(I.shape, dtype="bool")
    mask[0:l, :] = False
    mask[:, 0:l] = False
    mask[(I.shape[0] - l): I.shape[0], :] = False
    mask[:, (I.shape[0] - l): I.shape[0]] = False

    # create & fill array that contains the lookup for every possible
    # nxn template in the image
    idxArray = np.zeros((I[mask].size, n, n), dtype="int64")

    # compute the flattened indices for each position
    for i in range(0, n):
        for j in range(0, n):
            # idx contains the indices starting at the first valid template center position
            # and does a shift (I.shape[1] dependant) column wise for every j loop
            # idx builds up the idxArray that will contain all valid flattened template indices
            idx = (xIdx + (i - l)) + (yIdx + (j - l)) * I.shape[1]
            # mask idx and write it to idxArray
            # after the execution of all loop operations idxArray contains all flattened valid template indices
            # f.e. idxArray[0,:,:] = ([0..10],[600..610]),..,[6000..6010]) for (11x11 templates)
            idxArray[:, j, i] = idx[mask]

    # templates contains all valid nxn templates
    templates = I.ravel()[idxArray]

    return templates


def border_replicate(matrix, window):
    """
    this function replicates the border of a given image, such that (afterwards)
    a filter with a given window size can be applied to every pixel of
    the input image

    matrix          - image as a nxm numpy array
    window          - window size which refers to the necessary pixel width of
                      replication needed
    --------------------------------------------------
    matrixC         - image with replicated border
    """

    rand = int(window / 2 - 0.5)
    (zIm, sIm) = matrix.shape

    # insert rows
    i = 0
    matrixR = matrix
    while i < rand:
        matrixR = np.insert(matrixR, 0, values=matrix[0, :], axis=0)
        matrixR = np.insert(
            matrixR, zIm + 1, values=matrix[zIm - 1, :], axis=0)
        i += 1

    # insert columns
    (zNew, sNew) = matrixR.shape

    i = 0
    matrixC = matrixR
    while i < rand:
        matrixC = np.insert(matrixC, 0, values=matrixR[:, 0], axis=1)
        matrixC = np.insert(matrixC, sNew + 1,
                            values=matrixR[:, sNew - 1], axis=1)
        i += 1

    return matrixC


def rank_transform_rectified_images(rectData, win=15):
    """
    rect_data - dict containing rectified stereo-pair images in the form
                rect_data[i]['img_1_rect']
                rect_data[i]['img_2_rect']
    win       - kernelsize for rank transform
    """

    img1r = rectData["img1r"]
    img2r = rectData["img2r"]
    img1rRep = border_replicate(img1r, win)
    img2rRep = border_replicate(img2r, win)
    img1rRank = rank_transform(img1rRep, win)
    img2rRank = rank_transform(img2rRep, win)

    return np.uint8(img1rRank), np.uint8(img2rRank)


# rank transform for whole image with transform window size mxm
def rank_transform(arr, m):
    """
    arr: array with grayscale values that get examined in mxm windows
    return: array with size=size(arr-m) with rank transform values
    """

    row = arr.shape[0]
    col = arr.shape[1]

    if m % 2 == 0:
        raise Exception(
            "Rank transform can not be performed: no center pixel. m has to be odd."
        )

    r_vals = np.zeros((row - m + 1, col - m + 1))

    for i in range(row - m + 1):
        for j in range(col - m + 1):
            r_vals[i, j] = rank(arr[i: i + m, j: j + m])

    return r_vals


# rank transform
def rank(arr):
    """
    arr: array with grayscale values that get examined
    return: number of pixels in array that have smaller intensity value
            than the center pixel as integer
    """

    row = arr.shape[0]
    col = arr.shape[1]

    if row % 2 == 0 or col % 2 == 0:
        raise Exception(
            "Rank transform can not be performed: no center pixel. Array dimensions have to be odd."
        )

    r = 0
    c = arr[int((row - 1) / 2), int((row - 1) / 2)]

    # (arr<c) is boolean array, sum() counts number of true elements
    r = (arr < c).sum()

    return r
