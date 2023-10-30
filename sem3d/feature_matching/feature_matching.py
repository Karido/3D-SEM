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
#                   functions for feature matching in stereo images
#                   mostly based on open cv implementations
#
#                   SURF, SIFT or RootSIFT need an individual build of OpenCV
#                   or an older version, otherwise AKAZE is recommended
#                   pip install opencv-contrib-python==3.4.2.16
#                   """


# libraries
import numpy as np
import cv2 as cv
from scipy import stats
import copy as cp


def detect_features(pair, algorithm, thKp):
    """
    keypoint and descriptor creation

    thKp        - adjusts the most sensitive parameter of the selected
                  feature algorithm (float or int dependent on algorithm)
    algorithm   - string
    pair        - list with two images as entries (numpy arrays)

    --------------------------------------------------
    kp1, des1   - keypoints as cv2.keyPoint class KeyPoint
                  descriptors vice versa
    kp2, des2   - ...
    """

    img_1 = pair[0]
    img_2 = pair[1]

    # algorithms
    if algorithm == "SIFT":
        sift = cv.SIFT_create(nfeatures=0,
                              nOctaveLayers=3,
                              contrastThreshold=thKp,
                              edgeThreshold=15,
                              sigma=1.2,
                              )
        kp1 = sift.detect(img_1, None)
        kp1, des1 = sift.compute(img_1, kp1)
        kp2 = sift.detect(img_2, None)
        kp2, des2 = sift.compute(img_2, kp2)

    elif algorithm == "ROOTSIFT":
        sift = cv.SIFT_create(nfeatures=0,
                              nOctaveLayers=3,
                              contrastThreshold=thKp,
                              edgeThreshold=15,
                              sigma=1.2,
                              )
        kp1 = sift.detect(img_1, None)
        kp1, des1 = sift.compute(img_1, kp1)
        kp2 = sift.detect(img_2, None)
        kp2, des2 = sift.compute(img_2, kp2)

        # apply the Hellinger kernel by first L1-normalizing and taking the square-root
        eps = 1e-7
        des1 /= des1.sum(axis=1, keepdims=True) + eps
        des1 = np.sqrt(des1)
        des2 /= des2.sum(axis=1, keepdims=True) + eps
        des2 = np.sqrt(des2)

    elif algorithm == "SURF":
        surf = cv.xfeatures2d.SURF_create(
            hessianThreshold=thKp,
            nOctaves=3,
            nOctaveLayers=3,
            extended=True,
            upright=False,
        )
        kp1 = surf.detect(img_1, None)
        kp1, des1 = surf.compute(img_1, kp1)
        kp2 = surf.detect(img_2, None)
        kp2, des2 = surf.compute(img_2, kp2)

    elif algorithm == "KAZE":
        kaze = cv.KAZE_create(
            extended=True,
            upright=False,
            threshold=thKp,
            nOctaves=3,
            nOctaveLayers=3,
            diffusivity=cv.KAZE_DIFF_PM_G2,
        )
        kp1 = kaze.detect(img_1, None)
        kp1, des1 = kaze.compute(img_1, kp1)
        kp2 = kaze.detect(img_2, None)
        kp2, des2 = kaze.compute(img_2, kp2)

    elif algorithm == "ORB":
        orb = cv.ORB_create(
            nfeatures=thKp,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=4,
            scoreType=cv.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=2,
        )
        kp1 = orb.detect(img_1, None)
        kp1, des1 = orb.compute(img_1, kp1)
        kp2 = orb.detect(img_2, None)
        kp2, des2 = orb.compute(img_2, kp2)

    elif algorithm == "AKAZE":
        akaze = cv.AKAZE_create(
            descriptor_type=cv.AKAZE_DESCRIPTOR_MLDB,
            descriptor_size=0,
            descriptor_channels=3,
            threshold=thKp,
            nOctaves=3,
            nOctaveLayers=3,
            diffusivity=cv.KAZE_DIFF_WEICKERT,
        )
        kp1 = akaze.detect(img_1, None)
        kp1, des1 = akaze.compute(img_1, kp1)
        kp2 = akaze.detect(img_2, None)
        kp2, des2 = akaze.compute(img_2, kp2)

    elif algorithm == "BRISK":
        brisk = cv.BRISK_create(thresh=thKp, octaves=0, patternScale=1.2)
        kp1 = brisk.detect(img_1, None)
        kp1, des1 = brisk.compute(img_1, kp1)
        kp2 = brisk.detect(img_2, None)
        kp2, des2 = brisk.compute(img_2, kp2)

    else:
        raise Exception(
            "Choose an algorithm from AKAZE, BRISK, KAZE, ORB, RootSIFT, SIFT, SURF."
        )

    return kp1, des1, kp2, des2


###############################################################################
# MATCHING FUNCTIONS
###############################################################################


def brute_force_matching(desL, desR, kpL, kpR, dataType, th):
    """
    Brute Force matching algorithm - takes feature points, their descriptions and the type:

    RETURNS matched feature points from two images

    kp1, des1   -  keypoints as cv2.keyPoint class KeyPoint
                   descriptors vice versa
    kp2, des2   -  ...
    dataType    -  'standard'      - Euclidean norm with KNN
                   'binary'        - Hamming norm with KNN
                   'binaryCross'   - Euclidean norm with Cross Check - Matcher returns only those
                                     matches with value (i,j) such that i-th descriptor in set A
                                     has j-th descriptor in set B as the best match and vice-versa
                   'standardCross' - Hamming norm with Cross Check
    th          -  threshold/ratio used in Lowe's ratio test
    --------------------------------------------------
    srcPts, dstPts   - keypoints as numpy arrays that passed the ratio test
                       of the descriptor matching
    """

    if dataType == "binary":
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(desL, desR, k=2)
        return lowe_norm(matches, kpL, kpR, th)

    elif dataType == "binaryCross":
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        matches = bf.match(desL, desR)
        matches = sorted(matches, key=lambda x: x.distance)
        srcPts = np.asarray([kpL[m.queryIdx].pt for m in matches])
        dstPts = np.asarray([kpR[m.trainIdx].pt for m in matches])
        return srcPts, dstPts

    elif dataType == "binaryORB":
        bf = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck=False)
        matches = bf.knnMatch(desL, desR, k=2)
        return lowe_norm(matches, kpL, kpR, th)

    elif dataType == "standardCross":
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desL, desR)
        matches = sorted(matches, key=lambda x: x.distance)
        srcPts = np.asarray([kpL[m.queryIdx].pt for m in matches])
        dstPts = np.asarray([kpR[m.trainIdx].pt for m in matches])
        return srcPts, dstPts

    else:
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(desL, desR, k=2)
        return lowe_norm(matches, kpL, kpR, th)


def flann_matching(desL, desR, kpL, kpR, dataType, th):
    """
    FLANN matching algorithms
    (shows advantages in execution time for large feature numbers > 2k)
    KDTREE for vector description, LSH for binary description
    adjust settings for a trade-off between speed and accuracy


    kp1, des1   -  keypoints as cv2.keyPoint class KeyPoint
                   descriptors vice versa
    kp2, des2   -  ...
    dataType    -  'standard'      - Euclidean norm with KNN
                   'binary'        - Hamming norm with KNN
                   'binaryCross'   - Euclidean norm with Cross Check - Matcher returns only those
                                     matches with value (i,j) such that i-th descriptor in set A
                                     has j-th descriptor in set B as the best match and vice-versa
                   'standardCross' - Hamming norm with Cross Check
    th          -  threshold/ratio used in Lowe's ratio test
    --------------------------------------------------
    srcPts, dstPts   - keypoints as numpy arrays that passed the ratio test
                       of the descriptor matching
    """
    if dataType == "standard":
        FLANN_INDEX_KDTREE = 1
        indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

    elif dataType == "standardKM":
        FLANN_INDEX_KMEANS = 2
        CENTERS_RANDOM = 0
        indexParams = dict(
            algorithm=FLANN_INDEX_KMEANS,
            branching=32,
            iterations=11,
            centers_init=CENTERS_RANDOM,
            cb_index=0.2,
        )

    elif dataType == "standardComposite":
        FLANN_INDEX_COMPOSITE = 3
        CENTERS_RANDOM = 0
        indexParams = dict(
            algorithm=FLANN_INDEX_COMPOSITE,
            trees=4,
            branching=32,
            iterations=11,
            centers_init=CENTERS_RANDOM,
            cb_index=0.2,
        )

    elif dataType in ("binary", "binaryORB"):
        FLANN_INDEX_LSH = 6
        indexParams = dict(
            algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1
        )

    searchParams = dict(checks=40)
    flann = cv.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(desL, desR, k=2)

    return lowe_norm(matches, kpL, kpR, th)


def lowe_norm(matches, kpL, kpR, ratio=0.75):
    """
    apply Lowe's ratio test to discard bad matches

    matches          - list of lists containing DMatch objects (openCV)
    kpL, kpR         - list of key point objects (openCV)
    ratio            - threshold/ratio used to discard bad matches
    --------------------------------------------------
    srcPts, dstPts   - numpy array float 64
                       keypoints for which the descriptor matching passed
                       the ratio test
    """

    good = []

    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
        else:
            pass

    srcPts = np.asarray([kpL[m.queryIdx].pt for m in good])
    dstPts = np.asarray([kpR[m.trainIdx].pt for m in good])

    return srcPts, dstPts


def lowe_norm_pts(matches, kpL, kpR, ratio=0.75):
    """
    apply Lowe's ratio test to discard bad matches
    use if keypoints are not in the open cv format

    matches          - list of lists containing DMatch objects (openCV)
    kpL, kpR         - list of key points
    ratio            - threshold/ratio used to discard bad matches
    --------------------------------------------------
    srcPts, dstPts   - numpy array float 64
                       keypoints for which the descriptor matching passed
                       the ratio test
    """

    good = []

    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    srcPts = np.asarray([kpL[m.queryIdx] for m in good])
    dstPts = np.asarray([kpR[m.trainIdx] for m in good])

    return srcPts[:, 0:2], dstPts[:, 0:2], srcPts[:, 2], dstPts[:, 2]


###############################################################################
# MOTION CONSTRAINTS
###############################################################################


def v_motion_check(m, th=100):
    """
    checks the motion of matches between two stereo pair images
    and removes the ones that show |v disparity| > th

    m           - dict containing pts1 and pts2 as matches of a stereo pair
                  pts1, pts2: float64 array numMatches x 2
    config     - dict with set flags
    --------------------------------------------------
    m           - list of dicts containing the matched features as pts1 and pts2
                  after applying the motion constraints
    out1        - numpy float64 arrays, numOut x2
    out2        - numpy float64 arrays, numOut x2
    """

    out1 = []
    out2 = []
    dist = []
    distMask = []
    mArr1 = cp.copy(m["pts1"])
    mArr2 = cp.copy(m["pts2"])

    for i in range(len(mArr1)):
        dist.append(mArr1[i, 0:2][1] - mArr2[i, 0:2][1])

    distMask1 = dist < np.float32(th)
    distMask2 = dist > np.float32(-th)
    distMask = distMask1 * distMask2
    distMaskArr = np.asarray(distMask)
    out1.append(m["pts1"][~distMaskArr])
    out2.append(m["pts2"][~distMaskArr])
    m["pts1"] = m["pts1"][distMaskArr]
    m["pts2"] = m["pts2"][distMaskArr]

    return m, np.squeeze(np.asarray(out1), axis=0), np.squeeze(np.asarray(out2), axis=0)


def u_motion_check(m, th=100):
    """
    checks the motion of matches between two stereo pair images
    and removes the ones that show
    u disparity > (distMed+(medAd + th))
    or u disparity < (distMed-(medAd + th))

    m           - dict containing pts1 and pts2 as matches of a stereo pair
                  pts1, pts2: float64 array numMatches x 2
    config     - dict with set flags
    --------------------------------------------------
    m           - list of dicts containing the matched features as pts1 and pts2
                  after applying the motion constraints
    out1        - numpy float64 arrays, numOut x2
    out2        - numpy float64 arrays, numOut x2
    """

    out1 = []
    out2 = []
    dist = []
    distMask = []
    mArr1 = cp.copy(m["pts1"])
    mArr2 = cp.copy(m["pts2"])

    for i in range(len(mArr1)):
        dist.append(mArr1[i, 0:2][0] - mArr2[i, 0:2][0])

    medAd = stats.median_abs_deviation(dist)
    distMed = np.median(dist)
    distMask1 = dist < (distMed + (medAd + th))
    distMask2 = dist > (distMed - (medAd + th))
    distMask = distMask1 * distMask2
    distArrMask = np.asarray(distMask)
    out1.append(m["pts1"][~distArrMask])
    out2.append(m["pts2"][~distArrMask])
    m["pts1"] = m["pts1"][distArrMask]
    m["pts2"] = m["pts2"][distArrMask]

    return m, np.squeeze(np.asarray(out1), axis=0), np.squeeze(np.asarray(out2), axis=0)


def remove_multiple_matched_pts(pts1, pts2):
    """
    remove points that are matched multiple times,
    may be beneficial when number of outliers is high

    function is not applicable when ORB or FAST (OpenCV implementation)
    are used due to the insufficient keypoint accuracy of the OpenCV
    implementation which results in high numbers of points sharing the exact
    same u- or v-coordinates
    """

    # Create an empty list to store unique elements
    uniqueListU1 = []
    uniqueListU2 = []
    uniqueListV1 = []
    uniqueListV2 = []
    ptsList1 = list(pts1)
    ptsList2 = list(pts2)
    ptsOut1 = []
    ptsOut2 = []
    pts1Unique = []
    pts2Unique = []

    # Iterate over the original list and for each element
    # add it to uniqueList, if its not already there.
    for i in range(len(ptsList1)):
        pt1 = ptsList1[i]
        pt2 = ptsList2[i]
        if (pt1[0] in uniqueListU1 and pt1[1] in uniqueListV1) or (
            pt2[0] in uniqueListU2 and pt2[1] in uniqueListV2
        ):
            ptsOut1.append(pts1[i])
            ptsOut2.append(pts2[i])
        else:
            uniqueListU1.append(pt1[0])
            uniqueListU2.append(pt2[0])
            uniqueListV1.append(pt1[1])
            uniqueListV2.append(pt2[1])
            pts1Unique.append(pts1[i])
            pts2Unique.append(pts2[i])

    # Return the list of unique elements and removed points
    return (
        np.asarray(pts1Unique),
        np.asarray(pts2Unique),
        np.asarray(ptsOut1),
        np.asarray(ptsOut2),
    )


###############################################################################
# OPEN CV FUNCTIONS /// PARAMETER INFORMATION
###############################################################################


# SIFT algorithm, uses standard description or BRIEF description
# nfeatures (def. 0) - number of best features to retain
# nOctaveLayers (def. 3) - number of layers in each octave
# contrastThreshold (def. 0.04) - used to filter out weak features in low-contrast region (the larger, the less features)
# edgeThreshold (def. 10) - used to filter out edge-like features (the larger, the more features)
# sigma (def. 1.6) - sigma of Gaussian applied to the input image (by weak camera and soft lenses should be lower)

# def create_SIFT():
#     sift = cv.xfeatures2d.SIFT_create(nfeatures = 0, nOctaveLayers = 3, contrastThreshold = 0.04, edgeThreshold = 15, sigma = 1.2)
#     return sift


# SURF algorithm, uses standard description or BRIEF description
# hessianThreshold (def. 100) - Threshold of Hessian Detector (filter out KP with different greyscale), the larger, the less feature
# nOctaves (def. 4) - number of pyramid octaves
# nOctaveLayers (def. 3) - number of octave layers
# extended (def. False) - usual use smaller descriptor flag (64 bits), if true use extended descriptor flag (128 bits)
# upright (def. False) - when false, compute orientation, when true, do not compute orientation

# extended = true makes number of matched points almost twice smaller
# nOctaves has almost no influence

# def create_SURF():
#     surf = cv.xfeatures2d.SURF_create(hessianThreshold = 78.5, nOctaves = 3, nOctaveLayers = 3, extended = True, upright = False)
#     return surf


# FAST detection algorithm
# threshold (def. 10) - minimal brightness difference between potential KP and its neighborhood
# nonmaxSuppression (def. True) - filter out the adjacent keypoints
# type (def. cv.FAST_FEATURE_DETECTOR_TYPE_9_16) - size of neighborhood to analyse (other types: 5_8 and 7_12)
# setting nonmaxSuppresion False makes enormous difference in amount of KP, but ratio found KP to matched KP goes down

# def create_FAST():
#     fast = cv.FastFeatureDetector_create(threshold = 9, nonmaxSuppression = True, type = cv.FAST_FEATURE_DETECTOR_TYPE_9_16)
#     return fast


# ORB algorithm
# nfeatures (def. 500) - maximum number of features to retain
# scaleFactor (def. 1.2) - pyramid decimation ratio (by 2 - classical pyramid - lesser amount of matches, by 1 - more nlevels to cover scale range (pyramid becomes higher but slimmer))
# nlevels (def. 8) - number of pyramid levels
# edgeThreshold (def. 31) - size of border, where features aren't detected - should be the same like patchSize
# firstLevel (def. 0) - should stay 0
# WTA_K (def. 2) - number of points that produce oriented BRIEF descriptor (by usings 3 or 4 one need to change matching norm on HAMMING2 (Brute Force typ 4 and 5))
# scoreType (def. cv.ORB_HARRIS_SCORE) - score to rank the features (other possibility: cv.ORB_FAST_SCORE - fast is faster, but less stable)
# patchSize (def. 31)
# fastThreshold (def. 20) - like in FAST

# def create_ORB():
#     orb = cv.ORB_create(nfeatures = 10000, scaleFactor = 1.2, nlevels = 8, edgeThreshold = 31, firstLevel = 0, WTA_K = 4, scoreType = cv.ORB_HARRIS_SCORE, patchSize = 31, fastThreshold = 12)
#     return orb


# BRISK algorithm
# thresh (def. 30) - detection threshold
# octaves (def. 3) - detection octaves (0 to single scale)
# patternScale (def 1.0) - scale to sampling pattern

# def create_BRISK():
#     brisk = cv.BRISK_create(thresh = 11, octaves = 0, patternScale = 1.2)
#     return brisk


# Good Features To Track algorithm (Shi-Tomasi Corner Detector)
# maxCorners (def. 1000) - maximal number of keypoints to retain
# qualityLevel (def. 0.01) - minimal quality of keypoints (the bigger, the less KP)
# minDistance (def. 1) - minimal euclidean distance between keypoints
# blockSize (def. 3) - size of pixels neighborhood to compute a covariation matrix
# useHarrisDetector (def. False) - use Harris Corner Detector instead of Shi-Tomasi
# k (def. 0.04) - free parameter of detector (only by Harris)

# def create_GFTT():
#     gftt = cv.GFTTDetector_create(maxCorners = 13000, qualityLevel = 0.01, minDistance = 2, blockSize = 3, useHarrisDetector = False, k = 0)
#     return gftt


# BRIEF description algorithm
# bytes (def. 32) - size of descriptor, other values: 16 and 64 (for comparison, by SIFT 512 bytes and by SURF 256/512 bytes)
# useOrientation (def. True)

# def create_BRIEF():
#     brief = cv.xfeatures2d.BriefDescriptorExtractor_create(bytes = 64, use_orientation = True)
#     return brief


# FREAK algorithm
# orientationNormalized (def. True) - enabling normalization
# scaleNormalized (def. True) - enabling normalization
# patternScale (def. 22) - scaling of description pattern
# nOctaves (def. 4) - nummber of covered octaves

# def create_FREAK():
#     freak = cv.xfeatures2d.FREAK_create(orientationNormalized = True, scaleNormalized = True, patternScale = 23, nOctaves = 5)
#     return freak


# KAZE algorithm
# extended (def. False) - if True, descriptor has 128 Bytes (normally 64)
# upright (def. False) - if true, do not compute rotation
# threshold (def. 0.001) - by default value very low number of KP, should be much much lower
# nOctaves (def. 4) - similar like by SURF
# nOctaveLayers (def. 4) - similar like by SURF
# diffusivity (def. cv.KAZE_DIFF_PM_G2) - type of diffusivity (other possiblities: DIFF_PM_G1, DIFF_WEICKERT or DIFF_CHARBONNIER)
# best results by diffusivity PM_G2 (by PM_G1 and WEICKERT slightly more KP, but less matches, by CHARBONNIER much more KP and slightly more matches)

# def create_KAZE():
#     kaze = cv.KAZE_create(extended = True, upright = False, threshold = 0.001, nOctaves = 3, nOctaveLayers = 3, diffusivity = cv.KAZE_DIFF_PM_G2)
#     return kaze


# AKAZE algorithm
# descriptor_type (def. cv.AKAZE_DESCRIPTOR_MLDB) - there is also possibility to use KAZE descriptor - both have upright versions (without rotation)
# descriptor_size (def. 0) - size in bytes, if 0, then maximal
# descriptor_channels (def. 3)
# threshold (def. 0.001)
# nOctaves (def. 4)
# nOctaveLayers (def. 4)
# diffusivity (def. cv.KAZE_DIFF_PM_G2)

# def create_AKAZE():
#     akaze = cv.AKAZE_create(descriptor_type = cv.AKAZE_DESCRIPTOR_MLDB, descriptor_size = 0, descriptor_channels = 3, threshold = 0.0000215, nOctaves = 3, nOctaveLayers = 3, diffusivity = cv.KAZE_DIFF_WEICKERT)
#     return akaze
