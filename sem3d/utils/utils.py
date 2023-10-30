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
#                   functions for determining important geometric quantities of
#                   the affine epipolar geometry in stereo images
#                   """


# libraries
import numpy as np
from numpy import linalg as la
import sys


def homogenize(m):
    """
    homogenizes an array of point coordinates in the image plane
    input:  m         -   (nx2) or (2xn) array
    output: mh        -   (nx3) or (3xn) array with added ones same form as input array
    """
    # transpose marker initialized as 0
    T = 0
    # transpose point array if of the form (nx2)
    if np.shape(m)[1] == 2 or np.shape(m)[1] == 3:
        m = np.transpose(m)
        T = 1
    else:
        pass
    # add ones
    i = np.shape(m)[1]
    ones = np.ones((1, i))
    mh = np.vstack((m, ones))
    # if array was transposed retranspose
    if T == 1:
        mh = np.transpose(mh)
    else:
        pass

    return mh


def dehomogenize(mh):
    """
    INPUT: homogeneous point vector m - (3xn)

    first normalizes the vector to the form (x, y, 1)
    then removes the ones
    """
    # transpose marker initialized as 0
    T = 0
    # transpose point array if of the form (nx3)
    if np.shape(mh)[1] == 3 or np.shape(mh)[1] == 4:
        mh = np.transpose(mh)
        T = 1
    else:
        pass

    # check if array is x4 or x3
    j = np.shape(mh)[0]
    # number of entries
    i = np.shape(mh)[1]
    for n in range(i):
        mh[:, n] = mh[:, n] / mh[j-1, n]
    m = np.delete(mh, j-1, 0)

    # if array was transposed retranspose
    if T == 1:
        m = np.transpose(m)
    else:
        pass

    return m


def compute_rmse(distances):
    n = len(distances)
    rmse = np.sqrt((1/n) * np.sum((distances)**2))
    return rmse


def get_skew_sym(v):
    # creates a skew symmetric 3x3 matrix of a 3-vector
    S = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return S


def rank2(F):
    # enforces rank2 of a fundamental 3x3 matrix
    [U, D, VT] = la.svd(F)
    indexMin = np.argmin(D)
    D[indexMin] = 0
    diagD = np.diag(D)
    Fr2 = U.dot(diagD).dot(VT)
    return Fr2


def get_epipoles(F):
    """
    computes the epipoles (e1 - left image, e2 - right image)
    using a fundamental matrix F as input
    the epipoles are the left (e2T) and right (e1) null space of F
    """
    # 1. this solution uses la.eig:
    #    U1 = np.transpose(F).dot(F);
    #    [eval1, evec1] = la.eig(U1);
    #    index1 = np.argmin(eval1);
    #    e1 = evec1[:,index1]
    #    e1 = e1 / e1[2]
    #    U2 = F.dot(np.transpose(F));
    #    [eval2, evec2] = la.eig(U2);
    #    index2 = np.argmin(eval1);
    #    e2 = evec2[:,index2]
    #    e2 = e2 / e2[2]

    # 2. this solution just uses singular value decomposition and is
    # easier to understand
    [U, D, VT] = la.svd(F)
    # the vector given by the last column of V provides a solution
    # that minimizes ||F e1|| and can be called the left null space of F (e1)
    V = np.transpose(VT)
    e1 = V[:, -1]
    e1 = e1 / e1[2]
    # the vector given by the last row of UT provides a solution
    # that minimizes ||e2T F|| and can be called the right null space of F
    UT = np.transpose(U)
    e2 = UT[-1, :]
    e2 = e2 / e2[2]
    return e1, e2


def rotation_angles_arcsin(R):
    # extract the rotation angles from a rotation matrix

    y = -np.arcsin(R[2, 0])
    x = np.arctan2(R[2, 1]/np.cos(y), R[2, 2]/np.cos(y))
    z = np.arctan2(R[1, 0]/np.cos(y), R[0, 0]/np.cos(y))

    return x/np.pi*180, y/np.pi*180, z/np.pi*180


def rotation_angles_arctan(R):
    # extract the rotation angles from a rotation matrix

    x = np.arctan2(R[2, 1], R[2, 2])
    y = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    z = np.arctan2(R[1, 0], R[0, 0])

    return (x/np.pi*180, y/np.pi*180, z/np.pi*180)


def bucketing(pts1o, pts2o, size, nPts=800, nBuckets=8):
    """
    defines buckets (rectangles), which divide the image in 
    several regions, can be used to pick a number of spatially distributed
    points over the image:

    pick random points from the defined regions with the aim to 
    pick the same number of points from every bucket,
    iterating through the regions, starting top left

    pts1, pts2  - umhomogenized points in form (num x 2)
    size        - array [int, int], which describes size of left image
    nPts        - an integer describing how many points should be random got   
    nBuckets    - number of buckets: nBuckets * nBuckets
    --------------------------------------------------
    pts1, pts2  - (nPts x 2) containing homogenized 
                  coordinates of the bucketed random points
    """

    if nPts > len(pts1o):
        raise Exception(
            'The number of points to select is higher than the number of matches.')

    # compute vertical and horizontal size of small rectangles
    paceV = size[0] / nBuckets
    paceH = size[1] / nBuckets

    # compute the number of rectangles (buckets),
    # in which the 2D coordinates lie,
    # buckets are numbered from the left to the right
    # and from the top to the bottom,
    # the score array contains the numbers and enables to
    # determine the bucket each coordinate lies in
    score = nBuckets * (pts1o[:, 0] // paceV) + pts1o[:, 1] // paceH
    score = np.array(score).astype(np.int8)

    bucketedPts1 = [pts1o[score == sc] for sc in np.arange(np.max(score))]
    bucketedPts2 = [pts2o[score == sc] for sc in np.arange(np.max(score))]

    numPpb = np.asarray([len(bpts) for bpts in bucketedPts1])
    mask = numPpb != 0
    bucketedPts1 = np.asarray(bucketedPts1)[mask].tolist()
    bucketedPts2 = np.asarray(bucketedPts2)[mask].tolist()

    pts1 = list()
    pts2 = list()
    j = 0
    i = 0

    while i < nPts:

        if j == len(bucketedPts1):
            numPpb = np.asarray([len(bpts) for bpts in bucketedPts1])
            mask = numPpb != 0
            bucketedPts1 = np.asarray(bucketedPts1)[mask].tolist()
            bucketedPts2 = np.asarray(bucketedPts2)[mask].tolist()
            j = 0

        entryLen = len(bucketedPts1[j][:])
        selected = int(np.random.choice(np.arange(entryLen), 1))
        pts1.append(bucketedPts1[j][selected])
        pts2.append(bucketedPts2[j][selected])
        bucketedPts1[j] = np.asarray(bucketedPts1[j])[
            ~np.isin(np.arange(entryLen), selected)]
        bucketedPts2[j] = np.asarray(bucketedPts2[j])[
            ~np.isin(np.arange(entryLen), selected)]

        j = j+1
        i = len(pts1)

    pts1 = np.array(np.squeeze(pts1)[:, 0:2]).astype(np.float64)
    pts2 = np.array(np.squeeze(pts2)[:, 0:2]).astype(np.float64)
    pts1 = np.reshape(pts1, (nPts, 2))
    pts2 = np.reshape(pts2, (nPts, 2))

    return pts1, pts2


def bucketing_sac(pts1o, pts2o, size, nPts=800, nBuckets=8):
    """
    defines buckets (rectangles), which divide the image in 
    several regions, used if desired in the SAC routine:

    pick random points from randomly selected but different regions

    pts1, pts2  - umhomogenized points in form (num x 2)
    size        - array [int, int], which describes size of left image
    nPts        - an integer describing how many points should be random got   
    nBuckets    - number of buckets: nBuckets * nBuckets
    --------------------------------------------------
    pts1, pts2  - (nPts x 2) containing homogenized 
                  coordinates of the bucketed random points
    """

    if nPts > len(pts1o):
        raise Exception(
            'The number of points to select is higher than the number of matches.')

    # compute vertical and horizontal size of small rectangles
    paceV = size[0] / nBuckets
    paceH = size[1] / nBuckets

    # compute the number of rectangles (buckets),
    # in which the 2D coordinates lie,
    # buckets are numbered from the left to the right
    # and from the top to the bottom,
    # the score array contains the numbers and enables to
    # determine the bucket each coordinate lies in
    score = nBuckets * (pts1o[:, 0] // paceV) + pts1o[:, 1] // paceH
    score = np.array(score).astype(np.int8)

    bucketedPts1 = [pts1o[score == sc] for sc in np.arange(np.max(score))]
    bucketedPts2 = [pts2o[score == sc] for sc in np.arange(np.max(score))]

    numPpb = np.asarray([len(bpts) for bpts in bucketedPts1])
    mask = numPpb != 0
    bucketedPts1 = np.asarray(bucketedPts1, dtype=object)[mask].tolist()
    bucketedPts2 = np.asarray(bucketedPts2, dtype=object)[mask].tolist()

    pts1 = list()
    pts2 = list()
    i = 0
    idx = np.random.choice(np.arange(len(bucketedPts1)),
                           size=(nBuckets*nBuckets)*2)

    while i < nPts:

        # pick random bucket
        j = int(np.random.choice(idx, size=1))
        mask = idx != j
        idx = idx[mask]

        if len(bucketedPts1[j][:]) == 0:
            numPpb = np.asarray([len(bpts) for bpts in bucketedPts1])
            mask = numPpb != 0
            bucketedPts1 = np.asarray(bucketedPts1, dtype=object)[
                mask].tolist()
            bucketedPts2 = np.asarray(bucketedPts2, dtype=object)[
                mask].tolist()

        entryLen = len(bucketedPts1[j][:])
        selected = int(np.random.choice(np.arange(entryLen), 1))
        pts1.append(bucketedPts1[j][selected])
        pts2.append(bucketedPts2[j][selected])
        bucketedPts1[j] = np.asarray(bucketedPts1[j], dtype=object)[
            ~np.isin(np.arange(entryLen), selected)]
        bucketedPts2[j] = np.asarray(bucketedPts2[j], dtype=object)[
            ~np.isin(np.arange(entryLen), selected)]

        j = j+1
        i = len(pts1)

    pts1 = np.array(np.squeeze(pts1)[:, 0:2]).astype(np.float64)
    pts2 = np.array(np.squeeze(pts2)[:, 0:2]).astype(np.float64)
    pts1 = np.reshape(pts1, (nPts, 2))
    pts2 = np.reshape(pts2, (nPts, 2))

    return pts1, pts2


def compute_centroid(points):
    """
    input: points nx2 or nx3
    """

    length = points.shape[0]
    sumX = np.sum(points[:, 0])
    sumY = np.sum(points[:, 1])

    return sumX/length, sumY/length


def dense_matches_from_disparity(img, disp, rectData, img_base_right=False):
    '''
    computes the dense matches from the masked disparity and rectData which contains
    the rectifying transformations
    input: img       - an original image
           disp      - unpadded disparity map 
           rectData  - rectification_data list with dicts
           img_base_right - True, right image is the origin of disparity

    '''

    Ht = rectData['Ht']
    T1_right = rectData['T1']
    T2_left = rectData['T2']

    if img_base_right == True:
        # second image as base image for the dense matching
        m1, m2 = extract_dense_matches_right(img, Ht, T1_right, T2_left, disp)
    else:
        # first image as base image for the dense matching
        m1, m2 = extract_dense_matches_left(img, Ht, T1_right, T2_left, disp)

    return m1, m2


def extract_dense_matches_left(img1, Ht, T1, T2, disp):
    """
    Function:
            Get corresponding point pairs between unrectified img1 and img2
            from the correspondences in the rectified images
    Parameters:
            img_1: original left image
            img_2: original right image
            Ht, T1, T2: transformation matrix
            disp: disparity map
    Return:
            corresponding points in left unrectified image and right unrectified image, corresponding colors
    """

    r, c = img1.shape[0], img1.shape[1]

    image1Pt = [[[j, i, 1] for i in range(r)] for j in range(c)]
    image1Pt = np.array(image1Pt).reshape((r*c, 3, 1))

    newImageCoords = inverse_rectification(Ht, image1Pt, T1)

    dispPt = [[[disp[i, j], 0, 0] for i in range(r)] for j in range(c)]
    dispPtRs = np.array(dispPt).reshape((r*c, 3, 1))

    image2Pt = image1Pt - dispPtRs
    newImageCoords2 = inverse_rectification(Ht, image2Pt, T2)

    # mask the invalid disparity values (minimum of disparity in opencv sgbm)
    # where no matches were found
    mask = np.squeeze(dispPtRs[:, 0]) > np.min(disp)

    newImageCoords1masked = newImageCoords[mask]
    newImageCoords2masked = newImageCoords2[mask]
    return newImageCoords1masked[:, :, 0], newImageCoords2masked[:, :, 0]


def extract_dense_matches_right(img1, Ht, T1, T2, disp):
    """
    Function:
            Get corresponding point pairs between unrectified img1 and img2
            from the correspondences in the rectified images
    Parameters:
            img_1: original left image
            img_2: original right image
            Ht, T1, T2: transformation matrix
            disp: disparity map
    Return:
            corresponding points in left unrectified image and right unrectified image, corresponding colors
    """

    r, c = img1.shape[0], img1.shape[1]

    image2Pt = [[[j, i, 1] for i in range(r)] for j in range(c)]
    image2Pt = np.array(image2Pt).reshape((r*c, 3, 1))

    newImageCoords2 = inverse_rectification(Ht, image2Pt, T2)

    dispPt = [[[disp[i, j], 0, 0] for i in range(r)] for j in range(c)]
    dispPtRs = np.array(dispPt).reshape((r*c, 3, 1))

    image1Pt = image2Pt - dispPtRs
    newImageCoords1 = inverse_rectification(Ht, image1Pt, T1)

    # mask the invalid disparity values (minimum of disparity in opencv sgbm)
    # where no matches were found
    mask = np.squeeze(dispPtRs[:, 0]) > np.min(disp)

    newImageCoords1masked = newImageCoords1[mask]
    newImageCoords2masked = newImageCoords2[mask]
    return newImageCoords1masked[:, :, 0], newImageCoords2masked[:, :, 0]


def inverse_rectification(Ht, imageCoords, H):
    """
    Function:
            Calculate image coordinates of "corresponding points of rectified image" in original image
            This function is to transform the coordinates of an rectified
            image into its unrectified state
    Parameters:
            Ht: rotation matrix from the Quatsi Euclidean Rectification
            image_coord: image coordinates of points in rectified image
            Hï¼rotation matrix of camera around focus from the Quatsi Euclidean Rectification
    Return:
            image coordinates of points in unrectified images
    """
    newImageCoords = np.linalg.inv(H).dot(np.linalg.inv(Ht)) @ (imageCoords)

    for i, coords in enumerate(newImageCoords):
        newImageCoords[i] = coords/coords[2]
    return newImageCoords


def inhom_triangulation(pts, P):
    """
    pseudo inverse based linear triangulation method    
    (inhomogeneous method MVG)

    pts     - (2*m)xnumPoints array
    P       -   truncated camera matrices as vertically stacked array of m (2*3) matrices
    X       -  3 x m array containing 3D points
    """
    A = P
    b = pts

    # compute pseudo-inverse
    A_pinv = np.linalg.pinv(A)

    # solve equation system for all points
    X = A_pinv.dot(b)
    return X.T


def hom_triangulation(x1, x2, P1, P2, normalize=True):
    """
    svd-based linear triangulation method    
    (homogeneous method DLT, MVG page 312)

    x1  - (nx3) list of points within image 1
    x2  - (nx3) matching list of points within image 2
    P1  - camera matrix P1 for camera that took image 1
    P2  - camera matrix P2 for camera that took image 2

    X   - 3D-Points homogeneous coordinates
    """
    count, d1 = x1.shape
    count2, d2 = x2.shape

    if not count == count2:
        raise ValueError("x1 and x2 must have same number of points/rows")
    if d1 == 3:  # homogeneous coordinates
        for i in range(count):
            x1[i, :] /= x1[i, -1]

    if d2 == 3:  # homogeneous coordinates
        for i in range(count):
            x2[i, :] /= x2[i, -1]

    X = np.empty((count, 4), dtype=float)
    A = np.empty((4, 4))

    for i in range(count):
        A[0, :] = x1[i, 0] * P1[2, :] - P1[0, :]
        A[1, :] = x1[i, 1] * P1[2, :] - P1[1, :]
        A[2, :] = x2[i, 0] * P2[2, :] - P2[0, :]
        A[3, :] = x2[i, 1] * P2[2, :] - P2[1, :]

        u, s, vT = la.svd(A, True)
        X[i, :] = vT[-1, :]

    if normalize:
        for i in range(count):
            X[i, :] /= X[i, -1]

    return X


def closestRQ(R):
    # computes the nearest orthogonal matrix using taylor series approach
    I = np.identity(R.shape[0])
    Y = R.T@R - I
    Rq = R - R@Y@(I/2 - (3*Y/8) + (5*Y**2/16) - 35*Y**4/128)
    return Rq


def closestRQ_II(R):
    # computes the nearest orthogonal matrix using SVD
    U, S, VT = la.svd(R)
    I = np.eye(3)
    Rq = U@I@VT
    return Rq


def affine_transform_3D_lsq(destination, source):
    """
    computes the affine transformations between two 3D point sets
    input:   destination - nx3 point array, fixed point cloud
             source      - nx3 point array, transformed point cloud
    """
    # pad the data with ones (homogenize coordinates)
    # therefore the transformation matrix can do translations too
    if np.shape(destination)[1] != 3:
        if np.shape(destination)[0] == 3:
            destination = destination.T
        else:
            sys.exit('use nx3 point array as input')

    def pad(x): return np.hstack([x, np.ones((x.shape[0], 1))])
    def unpad(x): return x[:, :-1]
    X = pad(source)
    Y = pad(destination)
    # solve the least squares problem X * A = Y
    # to find the transformation matrix A
    A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    def transform(x): return unpad(np.dot(pad(x), A))
    error = np.abs(destination - transform(source))
    return A, error


def transform_affine(pts, A):
    """
    pts - nx3 input 
    transform the point clouds using an affine transformation A
    """
    def pad(x): return np.hstack([x, np.ones((x.shape[0], 1))])
    def unpad(x): return x[:, :-1]
    def transform(x): return unpad(np.dot(pad(x), A))
    return transform(pts)


def rigid_transform_3D_lsq(pB, pA):
    """
    computes the rigid transformations between two 3D point sets
    according to Arun et al.
    input:   destination - pB - 3xn point array, fixed point cloud
             source - pA      - 3xn point array, transformed point cloud
    """
    # get centroid of both point sets
    centroid_A = np.mean(pA, axis=1)
    centroid_B = np.mean(pB, axis=1)
    # center the point sets around the origin
    pA_centered = pA - np.tile(centroid_A, (np.shape(pA)[1], 1)).T
    pB_centered = pB - np.tile(centroid_B, (np.shape(pA)[1], 1)).T
    # dot is matrix multiplication for array
    H = pA_centered.dot(pB_centered.T)

    U, S, Vt = la.svd(H)

    R_lsq = Vt.T.dot(U.T)

    if la.det(R_lsq) < 0:
        # print("Reflection detected")
        Vt[:, 2] *= -1
        R_lsq = Vt.T.dot(U.T)

    t_lsq = -R_lsq.dot(centroid_A) + centroid_B.T
    t_lsq = np.reshape(t_lsq, (np.shape(t_lsq)[0], 1))
    pA_T = R_lsq.dot(pA) + np.tile(t_lsq, np.shape(pA)[1])
    # calculate the error
    error = pA_T - pB
    return R_lsq, t_lsq, pA_T, error


def transform_R_t(pts, Rt):
    """
    pts - 3xn input 
    transform the point clouds (rotation and translation)
    """
    # extract rotation and translation
    R = Rt[0]
    t = Rt[1]
    R_lsq = np.vstack((R, np.zeros((1, 3))))
    t_lsq = np.vstack((t, 1))
    pts_tp = R_lsq.dot(pts) + np.tile(t_lsq, np.shape(pts)[1])
    pts_T = dehomogenize(pts_tp)
    return pts_T


def write_point_cloud_txt(fn, verts, colors):
    """
    write point cloud as txt-file
    """
    verts = np.array(verts.reshape(-1, 3))
    if colors == None:
        vertsStack = np.squeeze(verts)
    else:
        colors = np.array(colors.reshape(-1, 3))
        vertsStack = np.squeeze([np.hstack([verts, colors])])
    # open -> wb means 'write and binary' On Unix systems (Linux, Mac OS X, etc.), 
    # binary mode does nothing - they treat text files the same way that any other files are
    # treated. On Windows, however, text files are written with slightly modified line endings. 
    # This causes a serious problem when dealing with actual binary files, like exe or jpg files. 
    # Therefore, when opening files which are not supposed to be text, even in Unix, you should 
    # use wb or rb. Use plain w or r only for text files.
    with open(fn, "wb"
              ) as f:
        if colors == None:
            np.savetxt(f, vertsStack, fmt="%f %f %f ")
        else:
            np.savetxt(f, vertsStack, fmt="%f %f %f %d %d %d ")
