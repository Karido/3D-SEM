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
#                   algorithms to compute the affine fundamental matrix
#                   robustly and perform a subsequent optimization
#                   """

# libraries
import numpy as np

# modules
import sem3d.epipolar_geometry.distances as dist
import sem3d.utils.utils as utils


def estimate_robust_affine_fundamental_matrix(pts1, pts2, numSamples, size, config):
    """
    computes the epiSets that contain the best F and the
    corresponding inliers and standard deviation of the residuals
    based on the chosen SAC method

    pts1, pts2      - umhomogenized points in form (num x 2)
    numSamples      - integer, number of models to be estimated
    size            - size of images
    config         - dict
    --------------------------------------------------
    setsIn          - list with inPts1, inPts2, Fin, cost, stdEstimate
                      (homogenized points in form (num x 3) )
                      (fundamental matrix in form (3 x 3) )
    """

    method = config["robustMethod"]

    # compute numSamples of models (fundamental matrices)
    Farr = tensor_affine_fundamental_matrix(
        pts1, pts2, numSamples, size, config)

    if method == "RANSAC":
        setsIn = ransac(Farr, pts1, pts2, config)

    elif method == "MSAC":
        setsIn = msac(Farr, pts1, pts2, config)

    elif method == "LMedS":
        setsIn = lmeds(Farr, pts1, pts2, config)

    elif method == "MLESAC":
        setsIn = mlesac(Farr, pts1, pts2, config)

    return setsIn


def tensor_affine_fundamental_matrix(pts1, pts2, numSamples, size, config):
    """
    estimates numSamples affine fundamental matrices
    for numSamples sets of point correspondences from the minimum
    configuration set of 4 matches via the affine gold standard algorithm

    pts1, pts2  - unhomogenized points in form (num x 2)
    numSamples  - integer, number of computed models
    buckets     - bool, enables/disables bucketing
    size        - array [int, int], describes the size of the images
                  only used if buckets == True
    --------------------------------------------------
    F           - (numSamples x 3 x 3) 3D array of fundamental matrices
    buckets     - list of dicts with entries pts1 and pts2,
                  entries are four matches each, chosen by the bucketing
    """

    # number of points
    numPoints = len(pts1)

    if config["bucketing"] == True:
        # if enabled, use bucketing technique to select the four matches from which F is computed
        points1 = np.empty((numSamples, 4, 2))
        points2 = np.empty((numSamples, 4, 2))

        for i in range(numSamples):
            # select samples of 4 matches, distributed over the image space
            points1[i], points2[i] = utils.bucketing_sac(pts1, pts2, size, 4)

    else:
        # if False, just get 4 random correspondences for each sample
        idx = []

        for i in range(int(numSamples)):
            idx.append(np.random.choice(range(numPoints), 4, replace=False))

        idx = np.reshape(np.asarray(idx), (int(numSamples), 4))
        points1 = pts1[idx]
        points2 = pts2[idx]

    # adjust the order of points according to Hartley and Zisserman MVG
    finalSets = np.dstack((points2[:, :, 0:2], points1[:, :, 0:2]))

    # compute centroids of the point sets in the 3D array
    n = np.shape(finalSets)[2]
    centroids = np.einsum("ijk->ik", finalSets) / n
    centroidsShaped = np.reshape((centroids), (int(numSamples), 1, n))
    centroidsShapedTiled = np.tile(centroidsShaped, (1, n, 1))

    # solve for F affine
    A = finalSets - centroidsShapedTiled
    U, s, VT = np.linalg.svd(A)
    f = VT[:, 3, :]

    # extract values and build F
    F = np.zeros((np.shape(f)[0], 3, 3))
    F[:, 0, 2] = f[:, 0]
    F[:, 1, 2] = f[:, 1]
    F[:, 2, 0] = f[:, 2]
    F[:, 2, 1] = f[:, 3]
    F[:, 2, 2] = -np.einsum("ij,ij->i", f, centroids)

    return F


def local_opt_affine(matches, epiSet, config):
    """
    iterative local optimization of the best set yielded by the SAC scheme,
    guided matching is performed using the newly determined epipolar geometry
    1) determine Fglobal for all inliers
    2) use Fglobal to determine new inliers from all matches based on
       the used cost function
    3) store Fglobal, new inliers and cost
    4) perform at least once and repeat until the cost does not improve further

    matches - dict containing the matches in the stereo pair
    epiSet  - dict containing F and corresponding inliers computed via
              the chosen SAC scheme
    config - config['estimationSigma']
              config['costFunction']
              config['fixedSigma']
              config['kappa']
              config['searchRange']
              config['robustMethod']
    --------------------------------------------------
    npts1i,     - inliers identified for each iteration
    npts2i
    Fi          - F computed for each iteration
    i           - max number of iterations
    rho         - cost for each iteration (dependent on chosen method)
    stdi        - std deviation after each iteration
    std         - robust estimate or config['fixedSigma']
    threshold   - threshold computed as (config['kappa']*std)**2
    """

    if config["robustMethod"] == "LMedS":
        return local_opt_lmeds(matches, epiSet, config)
    elif config["robustMethod"] == "RANSAC":
        return local_opt_ransac(matches, epiSet, config)
    elif config["robustMethod"] == "MSAC":
        return local_opt_msac(matches, epiSet, config)
    elif config["robustMethod"] == "MLESAC":
        return local_opt_mlesac(matches, epiSet, config)
    else:
        return local_opt_mlesac(matches, epiSet, config)


def ransac(Farr, pts1, pts2, config):
    """
    Random Sample Consensus

    Farr          - numSamples x 3 x 3 fundamental matrices
    pts1, pts2    - unhomogenized points in form (num x 2)
    nSets         - returned n best sets
    sigma         - used when standard deviation is fixed
    cf            - type of cost function:
                    Sampson distance
                    symmetric epipolar distance
    stdEstimation - adaptive or fixed
    stdFac        - factor which defines the interval of accepted Gaussian
                    errorArrs together with the estimated standard deviation
    --------------------------------------------------
    """

    nSets = config["nSets"]

    errorArr, std, threshold, _ = compute_costs_thresholds(
        pts1, pts2, Farr, config)

    # check for which points (inliers) cost function is smaller than threshold
    mask = errorArr < threshold

    # count number of inliers
    trueIndices = np.count_nonzero(mask, 1)

    # find the n models (n samples) with highest number of inliers
    bestSamples = np.argpartition(trueIndices, -nSets)[-nSets:]

    # get indices of inliers corresponding to the best n models
    goodIndicesSet = mask[bestSamples]

    # inlier sets
    inPts1 = []
    inPts2 = []
    Fin = []
    rhoIn = []

    # get the best n sets unsorted
    for i in range(nSets):
        inPts1.append(pts1[goodIndicesSet[i]])
        inPts2.append(pts2[goodIndicesSet[i]])
        Fin.append(Farr[bestSamples[i]])
        rhoIn.append(np.sum(goodIndicesSet[i]))

    # sort sets and put them in list
    stdArr = np.repeat(std, nSets)
    setsIn = sorted(
        zip(inPts1, inPts2, Fin, rhoIn, stdArr), key=lambda x: len(x[0]), reverse=True
    )
    setsIn.append(std)

    return setsIn


def lmeds(Farr, pts1, pts2, config):
    """
    Least Median of Squares
    """

    nSets = config["nSets"]

    # compute costs
    errorArr, std, threshold, _ = compute_costs_thresholds(
        pts1, pts2, Farr, config)
    rho = np.median(errorArr, 1)

    # find the nSets best sets (not sorted yet)
    bestSets = np.argpartition(rho, nSets)[:nSets]
    newMask = errorArr[bestSets] < threshold

    # inlier sets
    inPts1 = []
    inPts2 = []
    Fin = []
    rhoIn = []

    # get the corresponding inliers and fundamental matrices (sets not sorted yet)
    for i in range(nSets):
        inPts1.append(pts1[newMask[i]])
        inPts2.append(pts2[newMask[i]])
        Fin.append(Farr[bestSets[i]])
        rhoIn.append(rho[bestSets[i]])

    # add the estimated standard deviation to every set
    stdArr = np.repeat(std, nSets)

    # sort the sets and zip the lists
    setsIn = sorted(
        zip(inPts1, inPts2, Fin, rhoIn, stdArr), key=lambda x: x[3], reverse=False
    )

    return setsIn


def msac(Farr, pts1, pts2, config):
    """
    M-Estimator Sample Consensus
    """

    nSets = config["nSets"]

    errorArr, std, threshold, _ = compute_costs_thresholds(
        pts1, pts2, Farr, config)

    # check for which points (outliers) cost function is greater than threshold
    mask = errorArr >= threshold

    # compute cost, score of outliers = threshold, score of inliers = cost function
    errorArr[mask] = threshold
    rho = np.sum(errorArr, 1)

    # find the nSets best models with lowest costs
    bestSets = np.argpartition(rho, nSets)[:nSets]
    newMask = errorArr[bestSets] < threshold

    # inlier sets
    inPts1 = []
    inPts2 = []
    Fin = []
    rhoIn = []

    # get the best n sets unsorted
    for i in range(nSets):
        inPts1.append(pts1[newMask[i]])
        inPts2.append(pts2[newMask[i]])
        Fin.append(Farr[bestSets[i]])
        rhoIn.append(rho[bestSets[i]])

    stdArr = np.repeat(std, nSets)
    setsIn = sorted(
        zip(inPts1, inPts2, Fin, rhoIn, stdArr), key=lambda x: x[3], reverse=False
    )

    return setsIn


def mlesac(Farr, pts1, pts2, config):
    """
    Maximum Likelihood Sample Consensus
    """

    vDist = config["searchRange"]
    nSets = config["nSets"]

    errorArr, std, threshold, _ = compute_costs_thresholds(
        pts1, pts2, Farr, config)

    numMatches = len(pts1)

    # compute the mixture parameter
    gamma = max_expectation_gamma_robust(
        std, errorArr, numMatches, vDist, iters=5)

    # compute logarithmic probability function of mixed distributions for each point
    Pi = gamma * (1 / (np.sqrt(2 * np.pi) * std)) * \
        np.exp(-errorArr / (2 * std**2))
    Po = (1 - gamma) / vDist
    L = -np.log(Pi + Po)

    # compute errorArr function rho, which sums logs
    rho = np.sum(L, 1)
    bestSets = np.argpartition(rho, nSets)[:nSets]
    newMask = errorArr[bestSets] < threshold

    # inlier sets
    inPts1 = []
    inPts2 = []
    Fin = []
    rhoIn = []

    # get the best n sets unsorted
    for i in range(nSets):
        inPts1.append(pts1[newMask[i]])
        inPts2.append(pts2[newMask[i]])
        Fin.append(Farr[bestSets[i]])
        rhoIn.append(rho[bestSets[i]])

    stdArr = np.repeat(std, nSets)
    setsIn = sorted(
        zip(inPts1, inPts2, Fin, rhoIn, stdArr), key=lambda x: x[3], reverse=False
    )

    return setsIn


def compute_costs_thresholds(pts1, pts2, Farr, config):
    """
    helper function that computes the costs for the matches
    based on the chosen cost function and determines the thresholds
    to classify inliers using the given or robustly estimated
    standard deviation

    Farr        - 3D array in form (numSamples x 3 x 3),
                  which contains fundamental matrices for each sample
    pts1, pts2  - homogenized points in form (n x 3)
    --------------------------------------------------
    errorArr    - 2D array in form (numSamples x numPoints)
    std         - std deviation
    threshold   - computed threshold via (kappa*std)**2
    leastMedian - least median of the numSamples medians computed for each
                  error array
    """

    costFunction = config["costFunction"]

    numMatches = len(pts1)

    # homogenize all points
    arrOnes = np.ones((numMatches, 1))
    pts1 = np.concatenate((pts1, arrOnes), 1)
    pts2 = np.concatenate((pts2, arrOnes), 1)

    # compute costs
    if costFunction == "symEpi":
        errorArr = dist.robust_residual_distance(Farr, pts1, pts2)
    else:
        errorArr = dist.robust_sampson(Farr, pts1, pts2)

    # compute medians of cost function
    medians = np.median(errorArr, 1)

    # find smallest median
    idx = np.argmin(medians)
    leastMedian = medians[idx]
    # robust sigma estimation
    std, threshold = compute_standard_deviation(
        numMatches, leastMedian, config)

    return errorArr, std, threshold, leastMedian


def compute_standard_deviation(numMatches, leastMedian, config):
    """
    leastMedian - least median of all computed samples of the SAC scheme
    numMatches  - number of all matches
    config     - config['estimationSigma'] == 'adaptive':
                  computes the robust estimate of the standard deviation
                  according to Leroy and Rousseaux
                  else:
                  assigns the fixed value of the standard deviation
                  config['fixedSigma']
    --------------------------------------------------
    std         - robust estimate or config['fixedSigma']
    threshold   - threshold computed as (config['kappa']*std)**2
    """

    if config["stdEstimation"] == "adaptive":
        # length of the parameter vector is 4 due to the affine epipolar geometry
        lenPv = 4
        # formula according to Leroy and Rousseaux
        std = 1.4826 * (1 + 5 / (numMatches - lenPv)) * np.sqrt(leastMedian)
        threshold = (config["kappa"] * std) ** 2
    else:
        std = config["stdFixed"]
        threshold = (config["kappa"] * std) ** 2

    return std, threshold


def max_expectation_gamma(std, errorArr, numMatches, vDist, iters=5):
    """
    computes the mixture parameter gamma via the expectation maximization
    algorithm

    std         - std deviation of the residuals
    errorArrArr    - array of computed errorArr values
    numMatches  - number of matches
    v           - parameter defined by the search range/disparity/ px distance
                  in which a potential match can occur
    iters       - number of iterations
    --------------------------------------------------
    gamma       - mixture parameter gamma
    """

    # gamma is the start value of the mixture parameter
    gamma = 0.5

    # expectation maximization algorithm according to Torr: MLESAC
    # errorArr is already squared
    for j in range(iters):
        Pi = (
            gamma
            * (1 / (np.sqrt(2 * np.pi) * std))
            * np.exp(-errorArr / (2 * std**2))
        )
        Po = (1 - gamma) / vDist
        z = Pi / (Pi + Po)
        z_s = np.sum(z)
        gamma = z_s / numMatches

    return gamma


def max_expectation_gamma_robust(std, errorArr, numMatches, vDist, iters=5):
    """
    computes the mixture parameter gamma via the expectation maximization
    algorithm

    std         - estimated std deviation of the residuals
    errorArrArr - list of arrays containing the computed errorArr values
    numMatches  - number of all matches
    v           - parameter defined by the search range/disparity/ px distance
                  in which a potential match can occur
    iters       - number of iterations
    --------------------------------------------------
    gamma       - numSamples x 1 array of float64 mixture parameters gamma
    """

    # gamma is the start value of the mixture parameter
    gamma = 0.5

    # expectation maximization algorithm according to Torr: MLESAC
    # errorArr is already squared
    for j in range(iters):
        Pi = (
            gamma
            * (1 / (np.sqrt(2 * np.pi) * std))
            * np.exp(-errorArr / (2 * std**2))
        )
        Po = (1 - gamma) / vDist
        z = Pi / (Pi + Po)
        z_s = np.sum(z, 1)
        gamma = z_s / numMatches
        gamma = np.reshape(gamma, (len(errorArr), 1))

    return gamma


###########################################################################
#   LOCAL OPTIMIZATION / GUIDED MATCHING
###########################################################################


def local_opt_lmeds(matches, epiSet, config):
    costFunction = config["costFunction"]

    stdi = []
    stdi.append(epiSet["std"])
    i = 0
    Fi = []
    npts1i = []
    npts2i = []
    mediani = []

    # append the initial correspondences and fundamental matrix
    F0 = epiSet["F"]
    pts1 = epiSet["pts1"]
    pts2 = epiSet["pts2"]
    m1 = matches["pts1"]
    m2 = matches["pts2"]
    Fi.append(F0)
    npts1i.append(pts1[:, :2])
    npts2i.append(pts2[:, :2])
    errorArr = dist.geometric_distance(
        m1, m2, F0, costFunction, score="single")
    leastMedian = np.median(errorArr)
    mediani.append(leastMedian)

    while True:
        pp12 = np.hstack((npts1i[i][:, 0:2], npts2i[i][:, 0:2]))
        Fglob = single_affine_fundamental_matrix(pp12)

        # compute residuals
        errorArr = dist.geometric_distance(
            m1, m2, Fglob, costFunction, score="single")

        # compute LMedS
        leastMedian = np.median(errorArr)
        numMatches = len(m1)

        # compute standard deviation
        std, threshold = compute_standard_deviation(
            numMatches, leastMedian, config)

        # determine inliers and compute median
        mask = errorArr < threshold
        npts1 = m1[mask]
        npts2 = m2[mask]
        mediani.append(leastMedian)

        # break condition (after five iterations)
        if leastMedian >= mediani[i] and (i > 4):
            Fi.append(Fglob)
            npts1i.append(npts1)
            npts2i.append(npts2)
            stdi.append(std)
            i += 1
            break
        else:
            Fi.append(Fglob)
            npts1i.append(npts1)
            npts2i.append(npts2)
            stdi.append(std)
            i += 1

    return npts1i, npts2i, Fi, i, mediani, stdi


def local_opt_ransac(matches, epiSet, config):
    costFunction = config["costFunction"]

    stdi = []
    stdi.append(epiSet["std"])
    i = 0
    Fi = []
    npts1i = []
    npts2i = []
    lenPts = []

    # append the initial correspondences and fundamental matrix
    F0 = epiSet["F"]
    pts1 = epiSet["pts1"]
    pts2 = epiSet["pts2"]
    m1 = matches["pts1"]
    m2 = matches["pts2"]
    Fi.append(F0)
    npts1i.append(pts1[:, :2])
    npts2i.append(pts2[:, :2])
    lenPts.append(len(pts1))

    while True:
        leni = len(npts1i[i])
        pp12 = np.hstack((npts1i[i][:, 0:2], npts2i[i][:, 0:2]))
        Fglob = single_affine_fundamental_matrix(pp12)
        errorArr = dist.geometric_distance(
            m1, m2, Fglob, costFunction, score="single")

        # compute LMedS and number of matches
        leastMedian = np.median(errorArr)
        numMatches = len(m1)

        # compute standard deviation
        std, threshold = compute_standard_deviation(
            numMatches, leastMedian, config)

        # determine inliers and get cost
        mask = errorArr < threshold
        npts1 = m1[mask]
        npts2 = m2[mask]
        lenNew = len(npts1)
        lenPts.append(lenNew)

        # break condition (after five iterations)
        if lenNew <= leni and i > 4:
            Fi.append(Fglob)
            npts1i.append(npts1)
            npts2i.append(npts2)
            stdi.append(std)
            i += 1
            break
        else:
            Fi.append(Fglob)
            npts1i.append(npts1)
            npts2i.append(npts2)
            stdi.append(std)
            i += 1

    return npts1i, npts2i, Fi, i, lenPts, stdi


def local_opt_msac(matches, epiSet, config):
    costFunction = config["costFunction"]
    Tfac = config["kappa"]

    stdi = []
    stdi.append(epiSet["std"])
    i = 0
    Fi = []
    npts1i = []
    npts2i = []
    rhoi = []

    # append the initial correspondences and fundamental matrix
    F0 = epiSet["F"]
    pts1 = epiSet["pts1"]
    pts2 = epiSet["pts2"]
    m1 = matches["pts1"]
    m2 = matches["pts2"]

    # compute initial cost
    errorArr = dist.geometric_distance(
        pts1, pts2, F0, costFunction, score="single")
    rho0 = np.sum(errorArr) + (len(m1) - len(pts1)) * (Tfac * stdi[0]) ** 2
    Fi.append(F0)
    npts1i.append(pts1[:, :2])
    npts2i.append(pts2[:, :2])
    rhoi.append(rho0)

    while True:
        # compute Fa with all correspondences involved
        pp12 = np.hstack((npts1i[i][:, 0:2], npts2i[i][:, 0:2]))
        Fnew = single_affine_fundamental_matrix(pp12)

        # compute error distances
        errorArr = dist.geometric_distance(
            m1, m2, Fnew, costFunction, score="single")

        # compute LMedS and number of matches
        leastMedian = np.median(errorArr)
        numMatches = len(m1)

        # compute standard deviation
        std, threshold = compute_standard_deviation(
            numMatches, leastMedian, config)

        # compute cost function rho for each point:
        # cost of outliers = threshold, cost of inliers = errorArr
        mask = errorArr >= threshold
        errorArr[mask] = threshold
        rhoNew = np.sum(errorArr)

        # find inliers
        newMask = errorArr < threshold
        npts1 = m1[newMask]
        npts2 = m2[newMask]
        rhoi.append(rhoNew)

        # break condition (after five iterations)
        if rhoNew >= rhoi[i] and i > 4:
            Fi.append(Fnew)
            npts1i.append(npts1)
            npts2i.append(npts2)
            stdi.append(std)
            i += 1
            break
        else:
            Fi.append(Fnew)
            npts1i.append(npts1)
            npts2i.append(npts2)
            stdi.append(std)
            i += 1

    return npts1i, npts2i, Fi, i, rhoi, stdi


def local_opt_mlesac(matches, epiSet, config):
    costFunction = config["costFunction"]
    vDist = config["searchRange"]

    stdi = []
    stdi.append(epiSet["std"])
    i = 0
    i = 0
    Fi = []
    npts1i = []
    npts2i = []
    rhoi = []

    # append the initial correspondences and fundamental matrix
    F0 = epiSet["F"]
    pts1 = epiSet["pts1"]
    pts2 = epiSet["pts2"]
    m1 = matches["pts1"]
    m2 = matches["pts2"]
    std = stdi[0]

    # compute residuals
    errorArr = dist.geometric_distance(
        m1, m2, F0, costFunction, score="single")

    # number of matches
    numMatches = len(m1)

    # maximization expectation algorithm (5 iterations)
    gamma = max_expectation_gamma(std, errorArr, numMatches, vDist, iters=5)

    # compute logarithmic probability function of mixed distributions for each point
    Pi = gamma * (1 / (np.sqrt(2 * np.pi) * std)) * \
        np.exp(-errorArr / (2 * std**2))
    Po = (1 - gamma) / vDist
    L = -np.log(Pi + Po)

    # compute the cost function as summed logs
    rho0 = np.sum(L)
    Fi.append(F0)
    npts1i.append(pts1[:, :2])
    npts2i.append(pts2[:, :2])
    rhoi.append(rho0)

    while True:
        # compute Fa with all correspondences involved
        pp12 = np.hstack((npts1i[i][:, 0:2], npts2i[i][:, 0:2]))
        Fnew = single_affine_fundamental_matrix(pp12)

        # compute residuals
        errorArr = dist.geometric_distance(
            m1, m2, Fnew, costFunction, score="single")

        # compute LMedS and number of matches
        leastMedian = np.median(errorArr)
        numMatches = len(m1)

        # compute standard deviation
        std, threshold = compute_standard_deviation(
            numMatches, leastMedian, config)

        gamma = max_expectation_gamma(
            std, errorArr, numMatches, vDist, iters=5)

        # compute logarithmic probability function of mixed distributions for each point
        Pi = (gamma * (1 / (np.sqrt(2 * np.pi) * std))
              * np.exp(-errorArr / (2 * std**2)))
        Po = (1 - gamma) / vDist

        # natural logarithm
        L = -np.log(Pi + Po)

        # determine inliers
        mask = errorArr < threshold

        # compute cost
        rhoNew = np.sum(L)
        npts1 = m1[mask]
        npts2 = m2[mask]
        rhoi.append(rhoNew)

        # break condition (after five iterations)
        if rhoNew >= rhoi[i] and i > 4:
            Fi.append(Fnew)
            npts1i.append(npts1)
            npts2i.append(npts2)
            stdi.append(std)
            i += 1
            break
        else:
            Fi.append(Fnew)
            npts1i.append(npts1)
            npts2i.append(npts2)
            stdi.append(std)
            i += 1

    return npts1i, npts2i, Fi, i, rhoi, stdi


def single_affine_fundamental_matrix(matches):
    """
    estimates one affine fundamental matrix
    using the affine gold standard algorithm

    matches -   2D array of size nx4 containing the points (x1, y1, x2, y2, ..)
    --------------------------------------------------------
    F       -   the estimated affine fundamental matrix
    """

    # adjust the order of points as defined by Hartley and Zisserman
    X = matches[:, [2, 3, 0, 1]]

    # compute the centroid
    n = len(X)
    centroid = np.sum(X, axis=0) / n

    # compute the (n x 4) matrix A of the centered matches
    A = X - np.tile(centroid, (n, 1))

    # the solution vector is given by the singular vector corresponding to the smallest
    # singular value of matrix A which is the last row of matrix VT
    U, s, VT = np.linalg.svd(A)
    f = VT[-1, :]

    # reshape values to obtain the affine fundamental matrix accoring to Hartley and Zisserman MVG
    F = np.zeros((3, 3))
    F[0, 2] = f[0]
    F[1, 2] = f[1]
    F[2, 0] = f[2]
    F[2, 1] = f[3]
    F[2, 2] = -np.dot(f, centroid)

    return F
