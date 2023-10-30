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
#                   implementation of a weighted median filter
#                   """


# libraries
import numpy as np


# packages
import sem3d.utils.preprocessing as pre


def wmf(disp_map, window=3, sigma=25.5):
    # load the disparity map and set the minimum value to nan
    (r, c) = disp_map.shape
    min = np.min(disp_map)
    disp_map[disp_map == min] = np.nan

    # calculate the sections D of the disparity map and their corresponding weight matrices W
    D_and_W, window, weightType, sigma = weights_for_wmf(
        disp_map, "exp", window, sigma)

    # apply WMF
    print("COMPUTE WEIGHTED MEDIAN:")
    erg = []

    for i in range(0, len(D_and_W)):
        D = D_and_W[i][0]
        W = D_and_W[i][1]
        median = weighted_median(D, W)
        erg.append(median)

    erg = np.asarray(erg)

    erg = np.reshape(erg, (r, c))

    #    # plot the filtered disparity map
    #    import matplotlib.pyplot as plt
    #    plt.figure()
    #    plt.imshow(erg, cmap = 'jet')
    #    plt.title('disparitymap filtered with WMF')
    #    plt.text(5,r-5,'window size: %.0f x %.0f \nweighttype = %r \nsigma = %.1f'
    #             %(window,window,weightType,sigma),
    #             style='italic', fontsize = 8)

    return erg


def weights_for_wmf(matrix, weightType="exp", window=3, sigma=25.5):
    """
    input:  matrix               - unfiltered disparity map with minimum and maximum
                                   values set to nan. In order to avoid a long runtime of the code,
                                   the dispartiy map should be cropped to the area of interest
            weightType           - weight form that defines the pixel affinity of pixel
                                   "p" and "q"
                                   'exp': exp(-|F(p)-F(q)|^2/(2*sigma^2))
                                   'iv1': (|F(p)-F(q)|+sigma)^-1
                                   'iv2': (|F(p)-F(q)|^2+sigma^2)^-1
            window               - window radius, uneven number, minimum = 3
            sigma                - sigma controls the weight between two pixels
                                   range (0, inf)
    output: D                    - list that contains numpy arrays:
                                   each pixel of the matrix with its surrounding
                                   neighbour pixels within the given window size
                                   is contained in one numpy array
            W                    - list that contains numpy arrays:
                                   for each numpy array in D the corresponding weights
                                   are listed in W
                                   In D_and_W the first column contains the arrays of D
                                   and the second column the arrays of W
            z_D, s_D             - size of D
            window
            sigma
    """

    if window % 2 == 0:
        raise Exception(
            "Rank transform can not be performed: no center pixel. m has to be odd."
        )

    rand = int(window / 2 - 0.5)
    mitte = int(window * window / 2 - 0.5)

    # replicate the border of the matrix
    matrix = pre.border_replicate(matrix, window)
    (z, s) = matrix.shape

    # functions for the calculation of the weights
    def function_exp(i, j):
        return np.exp((-((np.absolute(i - j)) ** 2)) / (2 * sigma**2))

    def function_iv1(i, j):
        return (np.absolute(i - j) + sigma) ** (-1)

    def function_iv2(i, j):
        return ((np.absolute(i - j)) ** 2 + sigma**2) ** (-1)

    # initialise D and W
    D = []
    W = []
    for a in range(0, z - 2 * rand):
        for b in range(0, s - 2 * rand):
            D.append(matrix[a: a + 2 * rand + 1, b: b + 2 * rand + 1])

    for i in range(0, len(D)):
        D[i] = (D[i]).flatten()

    for i in range(0, len(D)):
        if D[i][mitte] != D[i][mitte]:
            weight = np.empty(len(D[i]))
            weight[:] = np.nan
            W.append(weight)

        else:
            if weightType == "iv1":
                W.append(function_iv1(D[i][:], D[i][mitte]))

            elif weightType == "iv2":
                W.append(function_iv2(D[i][:], D[i][mitte]))

            else:
                W.append(function_exp(D[i][:], D[i][mitte]))
                print("Default weightType exp is used.")

    # save the results for D and W in one list
    D_and_W = []
    pair = []
    for j in range(len(D)):
        pair.append(D[j])
        pair.append(W[j])
        D_and_W.append(pair)
        pair = []
    print("OK\n")
    return D_and_W, window, weightType, sigma


def weighted_median(D, W):
    """
    input:  D                    - numpy array containing a pixel of the matrix
                                   with its surrounding neighbour pixels
            W                    - numpy array:
                                   for the numpy array in D the corresponding weights
                                   are listed in W
    output: wMed                 - calculated weighted median
    """

    mitte = int(len(D) / 2 - 0.5)

    wMed = []

    if D[mitte] != D[mitte]:
        wMed = np.nan

    else:
        # test, if value of W is nan
        # values, that have been nan, are now zero
        W_ohne_nan = np.nan_to_num(W)

        A = np.vstack([D, W_ohne_nan])
        A = A.transpose()
        A_sort = A[A[:, 0].argsort()]

        # if a value of D is nan, the corresponding weight in W is set to zero
        ix = np.isnan(A_sort[:, 0])
        iy = np.where(ix)
        count_nan = len(iy[0])
        A_sort[len(A) - count_nan: len(A), 1] = 0

        # normalising of the weights
        W_sum = np.sum(A_sort[:, 1])
        A_sort[:, 1] = A_sort[:, 1] / W_sum

        #        # if more than half of the values of D is nan -> wMed = nan
        #        count_nan = np.count_nonzero(np.isnan(A_sort[:,0])) #number of nan-values of D
        #        if count_nan > len(A_sort[:,0])/2:
        #            wMed = np.nan

        # adding up the weights and calculating the weighted median
        sumVec = np.zeros(len(A) ** 2)
        sumVec[0] = A_sort[0, 1]
        i = 0
        while not wMed:
            i += 1
            sumVec[i] = A_sort[i, 1] + sumVec[i - 1]
            if sumVec[i] >= 0.5:
                wMed = A_sort[i, 0]
                break

    return wMed
