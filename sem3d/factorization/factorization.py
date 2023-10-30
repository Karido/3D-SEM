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
#                   factorization methods and related algorithms regarding
#                   the basic orthographic approach and the extended
#                   scaled orthographic approach as well as Quan's cost
#                   functions
#                   """


# libraries
import numpy as np
from scipy.optimize import least_squares

# packages
import sem3d.utils.utils as utils


def factorization_or(W, decomp="eigen"):
    """
    Factorization Method according to
    Shape and Motion from image streams, Tomasi and Kanade 1992

    Orthographic Factorization corresponds to the orthographic camera

    Input: W  -  measurement matrix containing point correspondences in
                 m frames
                 W = [u0,u1,...un,v0,v1,...,vn].T
                 list with m*2 entries containing n coordinates [mxn]
                 u and v coordinates of all points for all image frames 0...n
    """

    # number of frames
    nof = int(len(W[:, 0]) / 2)
    U, s, V = np.linalg.svd(W)

    # Measurement Matrix W has max. rank(W)=3 (rank theorem), thus 3 singular values
    # by using only the 3 highest eigenvalues, the best rank 3 estimate for a noisy measurement matrix W is obtained
    U = U[:, 0:3]
    s = s[0:3]

    # s^(1/2)
    s12 = np.diag(np.sqrt(s))
    V = V[0:3, :]
    R = U @ s12

    # orthographic motion/rotation matrix
    S = s12 @ V

    # build linear system of metric constraints Ax=b
    A = np.empty((0, 6))
    b = list()

    for i in range(0, nof * 2):
        row = [
            R[i, 0] ** 2,
            2 * R[i, 0] * R[i, 1],
            2 * R[i, 0] * R[i, 2],
            R[i, 1] ** 2,
            2 * R[i, 1] * R[i, 2],
            R[i, 2] ** 2,
        ]
        A = np.vstack((A, row))
        b.append(1)

    for i in range(0, nof):
        row = [
            R[i, 0] * R[i + nof, 0],
            R[i, 0] * R[i + nof, 1] + R[i, 1] * R[i + nof, 0],
            R[i, 0] * R[i + nof, 2] + R[i, 2] * R[i + nof, 0],
            R[i, 1] * R[i + nof, 1],
            R[i, 1] * R[i + nof, 2] + R[i, 2] * R[i + nof, 1],
            R[i, 2] * R[i + nof, 2],
        ]
        A = np.vstack((A, row))
        b.append(0)

    b = np.array(b).T

    # solve with Moore-Penrose Pseudoinverse
    I = np.linalg.pinv(A) @ b

    # build the symmetric matrix L
    L = np.array([[I[0], I[1], I[2]], [I[1], I[3], I[4]], [I[2], I[4], I[5]]])

    # eigendecompose L to compute the nearest positive definite matrix
    w, v = np.linalg.eig(L)

    # set negative eigenvalues to eps>0 to get closest positive definite matrix in frobenius norm
    w = w.clip(min=0)
    w[w == 0] = 10 ** (-9)

    if decomp == "cholesky":
        # use cholesky decomposition to obtain Q
        # get L positive definite
        Lpd = v @ np.diag(w) @ v.T
        Q = np.linalg.cholesky(Lpd)
    elif decomp == "eigen":
        # use eigen decomposition to obtain Q:
        # L = Q Q.T = U D U.T = U D^1/2*D^1/2 U.T -> Q = U D^1/2
        w = np.diag(np.sqrt(w))
        Q = v @ w

    # compute euclidean cameras and structure, upgrade with Q
    R = R @ Q
    S = np.linalg.inv(Q) @ S

    # construct one rotation matrix for each frame
    RotationMatrices = []
    for i in range(0, nof):
        RotationMatrices.append(
            np.array([R[i, :], R[i + nof, :], np.cross(R[i, :], R[i + nof, :])])
        )

    # set Frame 1 to world coordinate system -> first rotation matrix is identity matrix
    RotationToWorld = np.linalg.inv(RotationMatrices[0])
    for i in range(0, nof):
        RotationMatrices[i] = RotationMatrices[i] @ RotationToWorld
    Shape = S.T @ RotationToWorld

    ki = np.ones((nof))
    c = 1
    s = 0
    params = np.hstack((c, s))

    return RotationMatrices, Shape, ki, params, Q, s


def factorization_sc(W, decomp="eigen", normRot=0):
    """
    Factorization Method according to the Paper: Paraperspective Factorization
    of Poelman and Kanade, models the scaling effect of perspective projection
    when the camera moves in direction of the optical axis

    Input: W  -  measurement matrix containing point correspondences in
                 m frames
                 W = [u0,u1,...un,v0,v1,...,vn].T
                 list with m*2 entries containing n coordinates [mxn]
                 u and v coordinates of all points for all image frames 0...n
    """

    nof = int(len(W[:, 0]) / 2)
    U, s, V = np.linalg.svd(W)
    U = U[:, 0:3]
    s = s[0:3]
    s12 = np.diag(np.sqrt(s))
    V = V[0:3, :]
    R = U @ s12
    S = s12 @ V

    # build linear system of scaled orthographic constraints Ax=b
    iT = R[0:nof]
    jT = R[nof:]

    # build linear system of metric constraints Ax=b
    G = np.vstack((gT(iT, iT) - gT(jT, jT),
                  gT(iT, jT), gT1(iT[0, :], iT[0, :])))
    b = np.vstack((np.zeros((nof * 2, 1)), 1))
    b = np.squeeze(np.array(b).T)

    # solve with Moore-Penrose Pseudoinverse
    I = np.linalg.pinv(G) @ b

    # build the symmetric matrix L
    L = np.array([[I[0], I[1], I[2]], [I[1], I[3], I[4]], [I[2], I[4], I[5]]])

    # eigendecompose L to compute the nearest positive definite matrix
    w, v = np.linalg.eig(L)

    # set negative eigenvalues to eps>0 to get closest positive definite
    # matrix in frobenius norm
    w = w.clip(min=0)
    w[w == 0] = 10 ** (-9)

    if decomp == "cholesky":
        # use cholesky decomposition to obtain Q
        # get L positive definite
        Lpd = v @ np.diag(w) @ v.T
        Q = np.linalg.cholesky(Lpd)
    elif decomp == "eigen":
        # use eigen decomposition to obtain Q:
        # L = Q Q.T = U D U.T = U D^1/2*D^1/2 U.T -> Q = U D^1/2
        w = np.diag(np.sqrt(w))
        Q = v @ w

    # compute euclidean cameras and structure
    R = R @ Q

    # compute k and get the motion parameters
    ki = []
    for i in range(0, nof * 2):
        # get the true scaling factors for the images
        ki.append(
            np.linalg.norm(R[i])
        )  # would be zf in the paper by Poelman and Kanade zf.append(1/np.linalg.norm(R[i]) )
        R[i] = R[i] / np.linalg.norm(R[i])

    S = np.linalg.inv(Q) @ S

    # construct one rotation matrix for each frame
    RotationMatrices = []
    kiMean = []

    # computing the mean of the scaling factors per "row"
    # m1 and n1 should have same scaling, due to noise they differ slightly
    # therefore the mean is used
    for i in range(0, nof):
        RotationMatrices.append(
            np.array([R[i, :], R[i + nof, :], np.cross(R[i, :], R[i + nof, :])])
        )
        kiMean.append((ki[i] + ki[i + nof]) / 2)

    # averaged scaling factors
    kiNew = []
    for kii in kiMean:
        kiNew.append(kii / kiMean[normRot])

    # set Frame 1 to world coordinate system -> first rotation matrix is identity matrix
    RotationToWorld = np.linalg.inv(
        RotationMatrices[normRot])  # generally 0 = identiy
    for i in range(0, nof):
        RotationMatrices[i] = RotationMatrices[i] @ RotationToWorld

    Shape = S.T @ RotationToWorld
    c = 1
    s = 0
    params = np.hstack((c, s))

    return RotationMatrices, Shape, kiNew, params, Q, ki


def gT(a, b):
    """
    a - vector of size frames x 3
    b - vector of size frames x 3
    g.T- vector of size frames x 6
    """

    g = np.array(
        (
            [
                a[:, 0] * b[:, 0],
                a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0],
                a[:, 0] * b[:, 2] + a[:, 2] * b[:, 0],
                a[:, 1] * b[:, 1],
                a[:, 1] * b[:, 2] + a[:, 2] * b[:, 1],
                a[:, 2] * b[:, 2],
            ]
        )
    )

    return g.T


def gT1(a, b):
    """
    creation of the vector that is used for solving the quadratic equations
    based on the equation system from the papers Tomasi/Kanade and
    Poelman/Kanade
    """
    g1 = np.array(
        (
            [
                a[0] * b[0],
                a[0] * b[1] + a[1] * b[0],
                a[0] * b[2] + a[2] * b[0],
                a[1] * b[1],
                a[1] * b[2] + a[2] * b[1],
                a[2] * b[2],
            ]
        )
    )

    return g1.T


def build_measurement_matrix(matches):
    """
    iterates over all matches, if in previous and next image matches are found, the feature point is saved and compared to
    matches in next image-> Measurement matrix (only works for 2 matches (3 images) until now)

    input:  matches - matches between stereo images([image1,image2],[image2,image3],...)
                      list of n dict, containing 2 numpy arrays each
                      with image coordinates of (descriptor) matched
                      feature points, keys: pts_1 and pts_2

    output: measurement matrix with registered point correspondences

    """

    keysOutput = []
    for i in range(0, len(matches) + 1):
        keysOutput.append("pts" + str(i + 1))

    # remove feature points that are multiply matched
    # for k in range(len(matches)):
    #     matches[k]['pts1'],idx=np.unique(matches[k]['pts1'],return_index=True,axis=0)
    #     matches[k]['pts1']=matches[k]['pts1'][idx.argsort()]
    #     matches[k]['pts2']=matches[k]['pts2'][idx]
    #     matches[k]['pts2']=matches[k]['pts2'][idx.argsort()]
    #     matches[k]['pts2'],idx=np.unique(matches[k]['pts2'],return_index=True,axis=0)
    #     matches[k]['pts2']=matches[k]['pts2'][idx.argsort()]
    #     matches[k]['pts1']=matches[k]['pts1'][idx]
    #     matches[k]['pts1']=matches[k]['pts1'][idx.argsort()]

    newMatches = dict()
    newMatches["pts1"] = matches[0]["pts1"]
    newMatches["pts2"] = matches[0]["pts2"]

    for k in range(1, len(matches)):
        # sort each points based on product of x and y coordinate to make sure matches[0]['pts2'] and matches[1]['pts1'] are in same order
        idx = np.argsort(
            newMatches[keysOutput[k]][:, 0] * newMatches[keysOutput[k]][:, 1]
        )

        for i in range(0, k + 1):
            newMatches[keysOutput[i]] = newMatches[keysOutput[i]][idx]
        idx = np.argsort(matches[k]["pts1"][:, 0] * matches[k]["pts1"][:, 1])
        matches[k]["pts1"] = matches[k]["pts1"][idx]
        matches[k]["pts2"] = matches[k]["pts2"][idx]
        # only keep matches a<->b, b<->c,when point in b is in both matches
        mask11 = np.isin(matches[k]["pts1"][:, 0:1],
                         newMatches[keysOutput[k]][:, 0:1])
        mask12 = np.isin(matches[k]["pts1"][:, 1:2],
                         newMatches[keysOutput[k]][:, 1:2])
        mask21 = np.isin(newMatches[keysOutput[k]]
                         [:, 0:1], matches[k]["pts1"][:, 0:1])
        mask22 = np.isin(newMatches[keysOutput[k]]
                         [:, 1:2], matches[k]["pts1"][:, 1:2])
        mask1 = np.logical_and(mask11, mask12)
        mask2 = np.logical_and(mask21, mask22)
        newMatches[keysOutput[k + 1]] = matches[k]["pts2"][mask1[:, 0], 0:2]

        for u in range(0, k + 1):
            newMatches[keysOutput[u]
                       ] = newMatches[keysOutput[u]][mask2[:, 0], 0:2]

    # center the data
    newMatchesCentered = dict()
    center = []
    for key in keysOutput:
        centroid = utils.compute_centroid(newMatches[key])
        center.append(centroid)
        newMatchesCentered[key] = newMatches[key] - centroid

    # construct registered Measurement Matrix
    measurementMatrix = np.empty((0, len(newMatchesCentered) * 2))
    for i in range(0, len(newMatchesCentered["pts1"])):
        newColumn = []
        for key in keysOutput:
            newColumn.append(newMatchesCentered[key][i][0])
        for key in keysOutput:
            newColumn.append(newMatchesCentered[key][i][1])
        measurementMatrix = np.vstack((measurementMatrix, newColumn))

    return measurementMatrix.T, center, newMatches, newMatchesCentered


def find_equivalent_points(m1, m2):
    """
    # determines a mask, showing which row coordinates are equivalent in two
    # n x 3 arrays
    """

    idxList2 = []
    idxMask2 = []
    for mi in m2:
        # np.argwhere(np.all(m1 == mi, axis=0))
        idx = np.where((m1 == mi).all(axis=1))
        idxList2.append(idx)

        if idx[0].size > 0:
            idxMask2.append(True)
        else:
            idxMask2.append(False)

    idxList1 = []
    idxMask1 = []
    for mi in m1:
        # np.argwhere(np.all(m1 == mi, axis=0))
        idx = np.where((m2 == mi).all(axis=1))
        idxList1.append(idx)

        if idx[0].size > 0:
            idxMask1.append(True)
        else:
            idxMask1.append(False)

    return idxMask1, idxMask2, idxList1, idxList2


def build_measurement_matrix_2(matches):
    """
    slower function

    determines the measurement matrix from m-1 sets of point correspondences
    related to several stereo pair images

    determine the measurement matrix in that all point correspondences
    over several views are present

    m0---m1  #  m2---m3  #  m4---m5  #  m6--m7 (5 pictures) 8 arrays
    1-2         2-3         3-4         4-5


    m0---m1  #  m2---m3  #  m4---m5 (4 pictures) 6 arrays
    1-2         2-3         3-4

    m0--m1  #  m2--m3      (3 pictures)  4 arrays
    1-2        2-3
    """

    # arrays containing point matches
    m = []
    for e in matches:
        m.append(e["pts1"])
        m.append(e["pts2"])

    nImgs = int(len(m) / 2 + 1)

    # iterations for all transitions between the different stereo pairs
    for i in range(nImgs - 2):
        # number of arrays that must be processed
        nArr = 4 + i * 2

        # select the correct point arrays per iteration 0 -> 1 + 2, 1 -> 3 + 4,
        # 2 -> 5 + 6, usw.
        idxMask1, idxMask2, idxList1, idxList2 = find_equivalent_points(
            m[nArr - 2 - 1], m[nArr - 2]
        )

        tm = []
        for n in range(nArr):
            if n <= (nArr - 2 - 1):
                tm.append(m[n][idxMask1])
            else:
                tm.append(m[n][idxMask2])

        tms = []
        for n in range(nArr):
            if n <= (nArr - 2 - 1):
                tms.append(tm[n][tm[i * 2 + 1][:, 0].argsort(), :])
            else:
                tms.append(tm[n][tm[i * 2 + 2][:, 0].argsort(), :])

        for n in range(nArr):
            m[n] = tms[n]

    W = []
    i = 0
    j = 0
    for n in range(nImgs):
        if n <= 1:
            W.append(m[n].T[0])
        if n > 1:
            j = j + 1
            i = n + j
            W.append(m[i].T[0])

    i = 0
    j = 0
    for n in range(nImgs):
        if n <= 1:
            W.append(m[n].T[1])
        if n > 1:
            j = j + 1
            i = n + j
            W.append(m[i].T[1])

    Wr = []
    for n in range(nImgs * 2):
        Wr.append(W[n] - np.mean(W[n]))

    #    if plot == 1:
    #        vis.plot_matches(images[0],images[n_imgs-1], m[0], m[n_imgs-1], lines = 1, n = npoints)

    keysOut = []
    for i in range(len(matches) + 1):
        keysOut.append("pts_" + str(i + 1))

    mW = {}
    i = 0
    j = 0
    for n in range(nImgs):
        if n <= 1:
            mW[keysOut[n]] = m[n]
        if n > 1:
            j = j + 1
            i = n + j
            mW[keysOut[n]] = m[i]

    return np.asarray(W), np.asarray(Wr), mW


def reprojected_measurement_matrix(result):
    """
    # input: result of factorization
    # [0] euclidean rotations
    # [1] euclidean structure
    # [2] k-factors, c-factor, skew factor
    # reproject sparse 3D points and get reprojected W
    """
    ki = result[2]
    c, s = result[3]
    eucShape = np.asarray(result[1])
    eucRot = np.asarray(result[0])
    uPts = []
    vPts = []
    # reproject 3D points into every image with its own rotation matrix
    j = 0
    P_trunc = []
    for rot in eucRot:
        Ka = ki[j] * np.array([[1 * c, 0], [s, 1]], dtype=np.float64)
        rePts = Ka @ rot[0:2, :].dot(eucShape.T)
        P_trunc.append(Ka @ rot[0:2, :])
        uPts.append(rePts[0])
        vPts.append(rePts[1])
        j = j + 1
    # get new measurement matrix
    reW = np.vstack((np.asarray(uPts), np.asarray(vPts)))
    return reW, P_trunc


def remove_outliers_repro(W, reW, thFac):
    """
    # checks the mean reprojection error for every point over all views
    # and removes the outliers according to the std from 0 per x- and
    # y-coordinate, so that only measurements with err < thFac are accepted
    """
    # get error of all matches in all images
    allErr = signed_error(reW, W)
    oriDist = np.sqrt(allErr[0] ** 2 + allErr[1] ** 2)
    mask = oriDist < thFac
    numMatches = np.shape(W)[1]
    numImgs = int(len(mask) / numMatches)
    maskAll = mask[:numMatches]
    for i in range(numImgs - 1):
        i = i + 1
        maskAll = maskAll * mask[numMatches * i: numMatches * (i + 1)]
    idx = np.argwhere(maskAll == False)
    Wnew = W.copy()
    WreNew = reW.copy()
    Wnew = np.delete(Wnew, idx, 1)
    WreNew = np.delete(reW, idx, 1)
    return Wnew, WreNew


def signed_error(reW, W):
    """
    x- and y-error per point correspondence in all views

    W   - measurement matrix
    reW - reprojected points of the measurement matrix
    ---------------------
    err - the signed error (u- and v- component)
    """
    nof = int(np.shape(W)[0] / 2)
    reSignErr = reW - W
    errU = np.hstack(reSignErr[0:nof])
    errV = np.hstack(reSignErr[nof:])
    err = np.vstack((errU, errV))
    return err


###############################
# Quan's approaches
###############################


def cost_weak_perspective(zArr, *M):
    """
    # cost function for non-linear optimization
    # of Quan's constraints for the weak perspective camera
    """
    z1, z2, z3, z4, z5 = zArr

    Z = np.array([[z1, 0, 0], [z2, z3, 0], [z4, z5, 1]])

    return residuals_weak_perspective(Z, M)


def residuals_weak_perspective(Z, M):
    X = Z @ Z.T
    Mn = int(len(M) / 2) - 1
    costSummands = []

    for i in range(Mn):
        # assign new values for loop iteration
        miT = np.reshape(M[i], (1, 3))
        mi = np.reshape(M[i], (3, 1))
        miP1T = np.reshape(M[i + 1], (1, 3))
        miP1 = np.reshape(M[i + 1], (3, 1))
        niT = np.reshape(M[i + Mn + 1], (1, 3))
        ni = np.reshape(M[i + Mn + 1], (3, 1))
        niP1T = np.reshape(M[i + 1 + Mn + 1], (1, 3))
        niP1 = np.reshape(M[i + 1 + Mn + 1], (3, 1))
        # compute the cost
        min1 = (miT @ X @ mi) / (niT @ X @ ni)
        sub1 = (miP1T @ X @ miP1) / (niP1T @ X @ niP1)
        summand1 = (min1 - sub1) ** 2
        costSummands.append(np.squeeze(summand1))
        summand2 = (miT @ X @ ni) ** 2
        costSummands.append(np.squeeze(summand2))

    mnT = np.reshape(M[Mn], (1, 3))
    nn = np.reshape(M[Mn + Mn + 1], (3, 1))
    # für summe bis v anstatt nur bis v-1
    summandExt = (mnT @ X @ nn) ** 2
    costSummands.append(np.squeeze(summandExt))
    return costSummands


def cost_general_affine(zArr, *M):
    """
    # cost function for non-linear optimization
    # of Quan's constraints for the affine camera
    """
    z1, z2, z3, z4, z5 = zArr

    Z = np.array([[z1, 0, 0], [z2, z3, 0], [z4, z5, 1]])
    return residuals_general_affine(Z, M)


def residuals_general_affine(Z, M):
    """
    # original residual computation for non-linear optimization
    # of Quan's constraints for the affine camera
    """
    X = Z @ Z.T
    Mn = int(len(M) / 2) - 1
    costSummands = []

    for i in range(Mn):
        # assign new values for loop iteration
        miT = np.reshape(M[i], (1, 3))
        mi = np.reshape(M[i], (3, 1))
        miP1T = np.reshape(M[i + 1], (1, 3))
        miP1 = np.reshape(M[i + 1], (3, 1))
        niT = np.reshape(M[i + Mn + 1], (1, 3))
        ni = np.reshape(M[i + Mn + 1], (3, 1))
        niP1T = np.reshape(M[i + 1 + Mn + 1], (1, 3))
        niP1 = np.reshape(M[i + 1 + Mn + 1], (3, 1))
        # compute the cost
        min1 = (miT @ X @ mi) / (niT @ X @ ni)
        sub1 = (miP1T @ X @ miP1) / (niP1T @ X @ niP1)
        summand1 = (min1 - sub1) ** 2
        costSummands.append(np.squeeze(summand1))
        min2 = (miT @ X @ ni) / (niT @ X @ ni)
        sub2 = (miP1T @ X @ niP1) / (niP1T @ X @ niP1)
        summand2 = (min2 - sub2) ** 2
        costSummands.append(np.squeeze(summand2))
    return costSummands


def factorization_quan(W, Qc, ki, method="weak perspective"):
    """
    Factorization Method according to Quan's cost functions for a weak
    perspecitve camera and general affine camera,
    Quan defined the intrinsic parameters in the way of k[roh, 0][s, 1]

    Input: W  -  measurement matrix containing point correspondences in
                 m frames
                 W = [u0,u1,...um,v0,v1,...,vm].T
                 list with m*2 entries containing n coordinates [mxn]
                 u and v coordinates of all points for all image frames 0...n
    """
    # number of frames
    m = int(len(W[:, 0]) / 2)
    U, s, V = np.linalg.svd(W)
    U = U[:, 0:3]
    s = s[0:3]
    V = V[0:3, :]
    # s^(1/2)
    s12 = np.diag(np.sqrt(s))
    R = U @ s12
    # affine projection/rotation matrix
    M = R.copy()
    arg = []
    arg.append(M)
    for i in range(m):
        arg.append(ki[i])
    S = s12 @ V
    # ShapeAffine = S.copy()
    z_start = np.array([Qc[0, 0], Qc[1, 0], Qc[1, 1],
                       Qc[2, 0], Qc[2, 1]]) / Qc[2, 2]
    # bounds of the angles and the focal length, the focal length gets recomputed obtaining the minimized value
    # levenberg-mqrquardt to minimize the const function formulated by Quan
    if method == "affine":
        result = least_squares(cost_general_affine,
                               z_start,
                               method="lm",
                               max_nfev=1000,
                               ftol=1e-15,
                               xtol=1e-15,
                               args=(M),
                               verbose=0,
                               )
    else:
        result = least_squares(cost_weak_perspective,
                               z_start,
                               method="lm",
                               max_nfev=1000,
                               ftol=1e-15,
                               xtol=1e-15,
                               args=(M),
                               verbose=0,
                               )

    z = result.x

    z1, z2, z3, z4, z5 = z

    Z = np.array([[z1, 0, 0], [z2, z3, 0], [z4, z5, 1]])

    Aii = []
    kii = []

    for i in range(m):
        M0 = np.array((M[i], M[i + m]))
        M0Z = M0 @ Z
        # LQ decomposition using transposed matrices and QR decomposition
        q, r = np.linalg.qr(M0Z.T)
        # getting L
        A = np.transpose(r)
        rk = A[1, 1]
        A = A / rk
        A[0, 0] = np.abs(A[0, 0])
        Aii.append(A)
        kii.append(rk)

    if kii[0] < 1:
        for element in kii:
            element = -element
    kii_n = kii / kii[0]

    # select the reference view
    M0 = np.array((M[0], M[0 + m]))
    M0Z = M0 @ Z
    # LQ decomposition by using transposed matrices and QR decomposition
    q, r = np.linalg.qr(M0Z.T)
  
    # getting Q
    R0 = np.transpose(q)

    # orthonormality constraint of third row of R
    R0_3 = np.array([R0[0], R0[1], np.cross(R0[0], R0[1])])
    Q = Z @ R0_3.T
    R = R @ Q

    # in case of degenerated rotation matrices
    # for i in range(0,m*2):
    #     R[i] = R[i] / np.linalg.norm(R[i])

    Shape = np.linalg.inv(Q) @ S
    # construct one rotation matrix for each frame
    RotationMatrices = []
    for i in range(0, m):
        RotationMatrices.append(
            np.array([R[i, :], R[i + m, :], np.cross(R[i, :], R[i + m, :])])
        )

    c = []
    s = []

    for mat in Aii:
        c.append(mat[0, 0])
        s.append(mat[1, 0])
    c = np.mean(c)
    s = np.mean(s)

    params = np.hstack((c, s))
    # for WP factorization shape is scaled by the factors 1/ki
    # cause there is no constraint on overall scale, therefore
    # shape is rescaled with the factor that is related to k = 1
    Shape = Shape * kii[0]

    return RotationMatrices, Shape.T, kii_n, params, kii, Aii, Z, Q
