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
#                   functions for the computation of error-metrics with
#                   respect to the computation of the fundamental matrix
#                   from point correspondences
#                   """


# libraries
import numpy as np


def robust_sampson(Farr, pts1, pts2):
    """
    computes the sampson distances for each sample

    Farr        - 3D array in form (numSamples x 3 x 3),
                  which contains fundamental matrices for each sample
    pts1, pts2  - homogenized points in form (n x 3)
    --------------------------------------------------
    sampsonDist - 2D array in form (numSamples x numPoints)
    """

    # compute all neccessary terms
    a = np.einsum("ij,kjl->kil", pts2, Farr)
    a = np.einsum("ijk,jk->ij", a, pts1)
    asq = np.square(a)
    temp1 = np.einsum("ijk,lk->ijl", Farr, pts1)
    temp2 = np.einsum("ijk,lj->ikl", Farr, pts2)
    b = np.square(temp1[:, 0, :])
    c = np.square(temp1[:, 1, :])
    d = np.square(temp2[:, 0, :])
    e = np.square(temp2[:, 1, :])
    # calculate Sampson distance for each point for each sample
    sampsonDist = asq / (b + c + d + e)

    return sampsonDist


def robust_residual_distance(Farr, pts1, pts2):
    """
    computes the symmetrical epipolar distance for each sample,
    also referred to as residual error (Hartley and Zisserman)

    Farr         - 3D array in form (numSamples x 3 x 3),
                   which contains fundamental matrices for each sample
    pts1, pts2   - homogenized points in form (n x 3)
    --------------------------------------------------
    residualDist - 2D array in form (numSamples x numPoints)
    """

    # compute all neccessary terms
    a = np.einsum("ij,kjl->kil", pts2, Farr)
    a = np.einsum("ijk,jk->ij", a, pts1)
    asq = np.square(a)
    temp1 = np.einsum("ijk,lk->ijl", Farr, pts1)
    temp2 = np.einsum("ijk,lj->ikl", Farr, pts2)
    b = np.square(temp1[:, 0, :])
    c = np.square(temp1[:, 1, :])
    d = np.square(temp2[:, 0, :])
    e = np.square(temp2[:, 1, :])

    # compute residual errorArr for each point for each sample
    residualDist = asq * ((1 / (b + c)) + (1 / (d + e)))

    return residualDist


def residual_error(pts1, pts2, F):
    """
    computation of the distance between the point and the projected epoipolar
    line, squared and summed for both images which is also referred to as
    the symmetrical epipolar distance
    a different way to compute the symmetrical epipolar distance

    inputs:
    pts1 and pts2        - matrices of points in form (n x 3) or (n x 2)
    fundamental matrix F - in form (3 x 3)
    """

    # homogenize points, if it hasn't been done yet
    length = len(pts1)
    if len(pts1[0]) == 2:
        ON = np.ones((length, 1))
        pts1 = np.concatenate((pts1, ON), 1)
        pts2 = np.concatenate((pts2, ON), 1)

    # transpose matrices of points
    npts1 = np.einsum("ij->ji", pts1)
    npts2 = np.einsum("ij->ji", pts2)

    # compute the line on which the points are projected in the other image
    # by the fundamental matrix
    # epipolar line Ax + By + C = 0
    projectionLeft = np.einsum("ji,jk->ik", F, npts2)
    projectionRight = np.einsum("ij,jk->ik", F, npts1)
    Aleft = projectionLeft[0]
    Bleft = projectionLeft[1]
    Cleft = projectionLeft[2]
    Aright = projectionRight[0]
    Bright = projectionRight[1]
    Cright = projectionRight[2]

    # compute distances between points and epipolar lines
    distLeft = np.abs(Aleft * npts1[0] + Bleft * npts1[1] + Cleft) / np.sqrt(
        Aleft**2 + Bleft**2
    )
    distRight = np.abs(Aright * npts2[0] + Bright * npts2[1] + Cright) / np.sqrt(
        Aright**2 + Bright**2
    )

    # compute residual error = sum of both squared distances from one point's correspondence
    distLeft = np.square(distLeft)
    distRight = np.square(distRight)
    distance = distLeft + distRight

    # find maximal and average distance
    maxDistance = max(distance)
    avgDistance = np.mean(distance)
    rmseDistance = np.sqrt(np.sum(distance**2) / len(distance))

    return avgDistance, maxDistance, rmseDistance, distance


def sampson_distance(pts1, pts2, F):
    """
    compute Sampson distance for every point correspondence

    inputs:
    pts1 and pts2        - matrices of points in form (num x 3) or (num x 2)
    fundamental matrix F - (3 x 3)
    """

    distance = geometric_distance(pts1, pts2, F, "sampson", "single")
    maxDistance = max(distance)
    avgDistance = np.mean(distance)
    rmseDistance = np.sqrt(np.sum(distance**2) / len(distance))

    return avgDistance, maxDistance, rmseDistance, distance


def sym_epi_dist(pts1, pts2, F):
    """
    compute Sampson distance for every point correspondence

    inputs:
    pts1 and pts2        - matrices of points in form (num x 3) or (num x 2)
    fundamental matrix F - (3 x 3)
    """

    distance = geometric_distance(pts1, pts2, F, "symEpi", "single")
    maxDistance = max(distance)
    avgDistance = np.mean(distance)
    rmseDistance = np.sqrt(np.sum(distance**2) / len(distance))

    return avgDistance, maxDistance, rmseDistance, distance


def geometric_distance(pts1, pts2, F, costFunction, score):
    """
    inputs:
    pts1 and pts2        - matrices of points in form (num x 3) or (num x 2)
    fundamental matrix F - (3 x 3)
    score                - compute 'sum' or 'single values'
    costFunction         - 'Sampson' or 'symEpi'
    """

    if len(pts1[0]) == 2:
        length = len(pts1)
        ON = np.ones((length, 1))
        pts1 = np.concatenate((pts1, ON), 1)
        pts2 = np.concatenate((pts2, ON), 1)

    npts1 = np.einsum("ij->ji", pts1)
    npts2 = np.einsum("ij->ji", pts2)
    a = np.einsum("ji,jk->ik", npts2, F)
    a = np.einsum("ij,ji->i", a, npts1)
    a = np.square(a)
    temp1 = np.einsum("ij,jk->ik", F, npts1)
    temp2 = np.einsum("ji,jk->ik", F, npts2)
    b = np.square(temp1[0])
    c = np.square(temp1[1])
    d = np.square(temp2[0])
    e = np.square(temp2[1])

    if costFunction == "sampson":  # Sampson
        geoDist = a / (b + c + d + e)
    elif costFunction == "symEpi":  # symmetric epipolar distance
        geoDist = a * (1 / (b + c) + 1 / (d + e))

    if score == "sum":
        distance = np.einsum("i->", geoDist)
    elif score == "single":
        distance = geoDist

    return distance
