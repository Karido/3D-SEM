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
#                   implementation of uncalibrated rectification methods:

#                   general camera:
#                   "Quasi-euclidean epipolar rectification" of A. Fusiello and
#                   L. Irsara that uses point correspondences to rectify
#                   two images based on the pinhole camera model.

#                   affine cameras:
#                   Rectification method that exploits the parallelism of the
#                   epipolar lines to rectify the images directly from the
#                   entries of the affine fundamental matrix or the related
#                   point correspondences.
#                   """

# libraries
import numpy as np
from numpy.linalg import inv
import numpy.linalg as la
from scipy.optimize import least_squares
import cv2 as cv
import sys

# packages
import sem3d.utils.utils as utils
import sem3d.epipolar_geometry.distances as dist
import sem3d.visualization.output_visualization as vis


# ###############################################################################
#     QUASI EUCLIDEAN RECTIFICATION FUNCTIONS - START
# ###############################################################################


def euler_R(a):
    """
    eulR computes a rotation matrix given three euler angles
    rotations are applied in the order y-z-x (first y, last x)
    """
    # rotation angle around x
    xa = a[0]
    # rotation angle around y
    ya = a[1]
    # rotation angle around z
    za = a[2]
    # eigentlich Rz(alp) * Rx(bet) * Rz(gam), anscheinend hier egal
    Rx = np.array(
        ([1, 0, 0], [0, np.cos(xa), -np.sin(xa)], [0, np.sin(xa), np.cos(xa)])
    )
    Ry = np.array(
        ([np.cos(za), -np.sin(za), 0], [np.sin(za), np.cos(za), 0], [0, 0, 1])
    )
    Rz = np.array(
        ([np.cos(ya), 0, np.sin(ya)], [0, 1, 0], [-np.sin(ya), 0, np.cos(ya)])
    )
    R = Rx.dot(Ry).dot(Rz)
    # form of the final rotation matrix
    #    R2 = np.array(([np.cos(za)*np.cos(ya),      -np.sin(za),        np.cos(za)*np.sin(ya),
    #                   np.cos(xa)*np.sin(za)*np.cos(ya)+np.sin(xa)*np.sin(ya), np.cos(xa)*np.cos(za), np.cos(xa)*np.sin(za)*np.sin(ya)-np.sin(xa)*np.cos(ya),
    #                   np.sin(xa)*np.sin(za)*np.cos(ya)-np.cos(xa)*np.sin(ya), np.sin(xa)*np.cos(za), np.sin(xa)*np.sin(za)*np.sin(ya)+np.cos(xa)*np.cos(ya)]))
    return np.float64(R)


def qer_cost(a, w, h, ml, mr, focal):
    """
    compute the sampson error as a cost function
    a     -     is a vector of six elements containing the parameters
              five rotation angles
              (Y-left, Z-left, X-right, Y-right, Z-right and the value, which is
              further computed as (focal * a[4])*(w+h) to obtain the focal lenght.
    w,h   -     image width and height
    ml,mr -   feature matches
    """
    # recover parameters
    yl = a[0]
    zl = a[1]
    xr = a[2]
    yr = a[3]
    zr = a[4]
    # used to optimize the parameter a[5] in the interval [-1,1],
    # focal is user specified
    f = (focal ** a[5]) * (w + h)
    # estimate of the intrinsic parameters of the old cameras
    Kol = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    Kor = Kol
    # euler_R applies rotations in the order Y-Z-X
    Rl = euler_R(np.float64([0, yl, zl]))
    Rr = euler_R(np.float64([xr, yr, zr]))
    # get the fundamental matrix
    F = (
        np.transpose(inv(Kor))
        .dot(np.transpose(Rr))
        .dot(utils.get_skew_sym(np.float64([1, 0, 0])))
        .dot(Rl)
        .dot(inv(Kol))
    )
    if np.shape(mr)[1] != 3:
        m1 = utils.homogenize(ml)
        m2 = utils.homogenize(mr)
    else:
        m1 = ml
        m2 = mr
    u3 = utils.get_skew_sym([0, 0, 1])
    err = []
    for n in range(len(m1)):
        ufm1 = u3.dot(F).dot(m1[n, :])
        m2fu = np.transpose(m2[n, :]).dot(F).dot(u3)
        s = np.sqrt(
            np.square((np.transpose(m2[n, :]).dot(F).dot(m1[n, :])))
            / (np.transpose(ufm1).dot(ufm1) + m2fu.dot(m2fu.T))
        )
        err.append(s)
    return err


def qer_cost_fixed_focal(a, w, h, ml, mr, focal):
    """
    compute the sampson error as a cost function
    a     -   is a vector of five elements containing the parameters
              five rotation angles
              (Y-left, Z-left, X-right, Y-right, Z-right
    w,h   -   image width and height
    ml,mr -   feature matches
    """
    # recover parameters
    yl = a[0]
    zl = a[1]
    xr = a[2]
    yr = a[3]
    zr = a[4]
    # compute the fixed focal length from the user specified value focal
    f = focal * (w + h)
    # estimate of the intrinsic parameters of the old cameras
    Kol = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    Kor = Kol
    # euler_R applies rotations in the order Y-Z-X
    Rl = euler_R(np.float64([0, yl, zl]))
    Rr = euler_R(np.float64([xr, yr, zr]))
    # get the fundamental matrix
    F = (np.transpose(inv(Kor))
         .dot(np.transpose(Rr))
         .dot(utils.get_skew_sym(np.float64([1, 0, 0])))
         .dot(Rl)
         .dot(inv(Kol))
         )
    # print(F)
    if np.shape(mr)[1] != 3:
        m1 = utils.homogenize(ml)
        m2 = utils.homogenize(mr)
    else:
        m1 = ml
        m2 = mr
    u3 = utils.get_skew_sym([0, 0, 1])
    err = []
    for n in range(len(m1)):
        ufm1 = u3.dot(F).dot(m1[n, :])
        m2fu = np.transpose(m2[n, :]).dot(F).dot(u3)
        s = np.sqrt(
            np.square((np.transpose(m2[n, :]).dot(F).dot(m1[n, :])))
            / (np.transpose(ufm1).dot(ufm1) + m2fu.dot(m2fu.T))
        )
        err.append(s)
    return err


# MANUAL COMPUTATION OF THE JACOBIAN, NOT NECESSARY, APPROXIMATION OF scipy.least_squares IS SUFFICIENT
#
#
#    wph = w+h;
#    u0 = w/2;
#    v0 = h/2;
#    a6 = a[5]
#    # vector of F
#    f1 = F[0,0]; f4 = F[0,1]; f7 = F[0,2];
#    f2 = F[1,0]; f5 = F[1,1]; f8 = F[1,2];
#    f3 = F[2,0]; f6 = F[2,1]; f9 = F[2,2];
#    jac = []
#    for n in range(len(m1)):
#        m11 = m1[n,0];
#        m12 = m1[n,1];
#        m21 = m2[n,0];
#        m22 = m2[n,1];
#
#        df = [ \
#        [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  (-1/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.sin(zl)*np.sin(yl)+1/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))*np.cos(yl))/(3**a6)/wph,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           1/(3**a6)**2/wph**2*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.cos(zl)*np.cos(yl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  (1/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))*np.sin(zl)*np.cos(yl)+1/(3**a6)/wph*(-np.sin(xr)*np.sin(zr)*np.cos(yr)+np.cos(xr)*np.sin(yr))*np.sin(yl))/(3**a6)/wph,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 (1/(3**a6)/wph*(-np.sin(xr)*np.sin(zr)*np.sin(yr)-np.cos(xr)*np.cos(yr))*np.sin(zl)*np.cos(yl)+1/(3**a6)/wph*(-np.cos(xr)*np.sin(zr)*np.sin(yr)+np.sin(xr)*np.cos(yr))*np.sin(yl))/(3**a6)/wph,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       (1/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.cos(yr)*np.sin(zl)*np.cos(yl)+1/(3**a6)/wph*np.cos(xr)*np.cos(zr)*np.cos(yr)*np.sin(yl))/(3**a6)/wph,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          (-1/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.sin(zl)*np.cos(yl)*np.log(3)-1/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))*np.sin(yl)*np.log(3))/(3**a6)/wph-(1/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.sin(zl)*np.cos(yl)+1/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))*np.sin(yl))/(3**a6)/wph*np.log(3)], \
#        [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 (-1/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.sin(zl)*np.sin(yl)+1/(3**a6)/wph*np.cos(xr)*np.cos(zr)*np.cos(yl))/(3**a6)/wph,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     1/(3**a6)**2/wph**2*np.sin(xr)*np.cos(zr)*np.cos(zl)*np.cos(yl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       (1/(3**a6)/wph*np.cos(xr)*np.cos(zr)*np.sin(zl)*np.cos(yl)-1/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.sin(yl))/(3**a6)/wph,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      (-1/(3**a6)/wph*np.sin(xr)*np.sin(zr)*np.sin(zl)*np.cos(yl)-1/(3**a6)/wph*np.cos(xr)*np.sin(zr)*np.sin(yl))/(3**a6)/wph,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  (-1/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.sin(zl)*np.cos(yl)*np.log(3)-1/(3**a6)/wph*np.cos(xr)*np.cos(zr)*np.sin(yl)*np.log(3))/(3**a6)/wph-(1/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.sin(zl)*np.cos(yl)+1/(3**a6)/wph*np.cos(xr)*np.cos(zr)*np.sin(yl))/(3**a6)/wph*np.log(3)], \
#        [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       (-(-u0/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)+np.sin(xr)*np.sin(zr)*np.sin(yr)+np.cos(xr)*np.cos(yr))*np.sin(zl)*np.sin(yl)-(u0/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))+1/(3**a6)/wph*v0*np.cos(xr)*np.cos(zr)-np.cos(xr)*np.sin(zr)*np.sin(yr)+np.sin(xr)*np.cos(yr))*np.cos(yl))/(3**a6)/wph,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        (-u0/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)+np.sin(xr)*np.sin(zr)*np.sin(yr)+np.cos(xr)*np.cos(yr))*np.cos(zl)*np.cos(yl)/(3**a6)/wph,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           ((-u0/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.cos(xr)*np.cos(zr)+np.cos(xr)*np.sin(zr)*np.sin(yr)-np.sin(xr)*np.cos(yr))*np.sin(zl)*np.cos(yl)-(u0/(3**a6)/wph*(-np.sin(xr)*np.sin(zr)*np.cos(yr)+np.cos(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)+np.sin(xr)*np.sin(zr)*np.sin(yr)+np.cos(xr)*np.cos(yr))*np.sin(yl))/(3**a6)/wph,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          ((-u0/(3**a6)/wph*(-np.sin(xr)*np.sin(zr)*np.sin(yr)-np.cos(xr)*np.cos(yr))+np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.sin(zl)*np.cos(yl)-(u0/(3**a6)/wph*(-np.cos(xr)*np.sin(zr)*np.sin(yr)+np.sin(xr)*np.cos(yr))-np.cos(xr)*np.sin(zr)*np.cos(yr)-np.sin(xr)*np.sin(yr))*np.sin(yl))/(3**a6)/wph,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ((-u0/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.cos(yr)+1/(3**a6)/wph*v0*np.sin(xr)*np.sin(zr)+np.sin(xr)*np.cos(zr)*np.sin(yr))*np.sin(zl)*np.cos(yl)-(u0/(3**a6)/wph*np.cos(xr)*np.cos(zr)*np.cos(yr)-1/(3**a6)/wph*v0*np.cos(xr)*np.sin(zr)-np.cos(xr)*np.cos(zr)*np.sin(yr))*np.sin(yl))/(3**a6)/wph,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               ((u0/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.log(3)+1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)*np.log(3))*np.sin(zl)*np.cos(yl)-(-u0/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))*np.log(3)-1/(3**a6)/wph*v0*np.cos(xr)*np.cos(zr)*np.log(3))*np.sin(yl))/(3**a6)/wph-((-u0/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)+np.sin(xr)*np.sin(zr)*np.sin(yr)+np.cos(xr)*np.cos(yr))*np.sin(zl)*np.cos(yl)-(u0/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))+1/(3**a6)/wph*v0*np.cos(xr)*np.cos(zr)-np.cos(xr)*np.sin(zr)*np.sin(yr)+np.sin(xr)*np.cos(yr))*np.sin(yl))/(3**a6)/wph*np.log(3)], \
#        [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  -1/(3**a6)**2/wph**2*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.sin(zl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   1/(3**a6)**2/wph**2*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))*np.cos(zl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  1/(3**a6)**2/wph**2*(-np.sin(xr)*np.sin(zr)*np.sin(yr)-np.cos(xr)*np.cos(yr))*np.cos(zl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     1/(3**a6)**2/wph**2*np.sin(xr)*np.cos(zr)*np.cos(yr)*np.cos(zl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           -2/(3**a6)**2/wph**2*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.cos(zl)*np.log(3)], \
#        [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            -1/(3**a6)**2/wph**2*np.sin(xr)*np.cos(zr)*np.sin(zl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             1/(3**a6)**2/wph**2*np.cos(xr)*np.cos(zr)*np.cos(zl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            -1/(3**a6)**2/wph**2*np.sin(xr)*np.sin(zr)*np.cos(zl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     -2/(3**a6)**2/wph**2*np.sin(xr)*np.cos(zr)*np.cos(zl)*np.log(3)], \
#        [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               -(-u0/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)+np.sin(xr)*np.sin(zr)*np.sin(yr)+np.cos(xr)*np.cos(yr))*np.sin(zl)/(3**a6)/wph,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                (-u0/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.cos(xr)*np.cos(zr)+np.cos(xr)*np.sin(zr)*np.sin(yr)-np.sin(xr)*np.cos(yr))*np.cos(zl)/(3**a6)/wph,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               (-u0/(3**a6)/wph*(-np.sin(xr)*np.sin(zr)*np.sin(yr)-np.cos(xr)*np.cos(yr))+np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.cos(zl)/(3**a6)/wph,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  (-u0/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.cos(yr)+1/(3**a6)/wph*v0*np.sin(xr)*np.sin(zr)+np.sin(xr)*np.cos(zr)*np.sin(yr))*np.cos(zl)/(3**a6)/wph,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              (u0/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.log(3)+1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)*np.log(3))*np.cos(zl)/(3**a6)/wph-(-u0/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)+np.sin(xr)*np.sin(zr)*np.sin(yr)+np.cos(xr)*np.cos(yr))*np.cos(zl)/(3**a6)/wph*np.log(3)], \
#        [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    -(-1/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.sin(zl)*np.sin(yl)+1/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))*np.cos(yl))*u0/(3**a6)/wph+1/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.sin(zl)*np.cos(yl)+1/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))*np.sin(yl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          -1/(3**a6)**2/wph**2*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.cos(zl)*np.cos(yl)*u0+1/(3**a6)**2/wph**2*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.sin(zl)*v0+1/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.cos(zl)*np.sin(yl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 -(1/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))*np.sin(zl)*np.cos(yl)+1/(3**a6)/wph*(-np.sin(xr)*np.sin(zr)*np.cos(yr)+np.cos(xr)*np.sin(yr))*np.sin(yl))*u0/(3**a6)/wph-1/(3**a6)**2/wph**2*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))*np.cos(zl)*v0+1/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))*np.sin(zl)*np.sin(yl)-1/(3**a6)/wph*(-np.sin(xr)*np.sin(zr)*np.cos(yr)+np.cos(xr)*np.sin(yr))*np.cos(yl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              -(1/(3**a6)/wph*(-np.sin(xr)*np.sin(zr)*np.sin(yr)-np.cos(xr)*np.cos(yr))*np.sin(zl)*np.cos(yl)+1/(3**a6)/wph*(-np.cos(xr)*np.sin(zr)*np.sin(yr)+np.sin(xr)*np.cos(yr))*np.sin(yl))*u0/(3**a6)/wph-1/(3**a6)**2/wph**2*(-np.sin(xr)*np.sin(zr)*np.sin(yr)-np.cos(xr)*np.cos(yr))*np.cos(zl)*v0+1/(3**a6)/wph*(-np.sin(xr)*np.sin(zr)*np.sin(yr)-np.cos(xr)*np.cos(yr))*np.sin(zl)*np.sin(yl)-1/(3**a6)/wph*(-np.cos(xr)*np.sin(zr)*np.sin(yr)+np.sin(xr)*np.cos(yr))*np.cos(yl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             -(1/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.cos(yr)*np.sin(zl)*np.cos(yl)+1/(3**a6)/wph*np.cos(xr)*np.cos(zr)*np.cos(yr)*np.sin(yl))*u0/(3**a6)/wph-1/(3**a6)**2/wph**2*np.sin(xr)*np.cos(zr)*np.cos(yr)*np.cos(zl)*v0+1/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.cos(yr)*np.sin(zl)*np.sin(yl)-1/(3**a6)/wph*np.cos(xr)*np.cos(zr)*np.cos(yr)*np.cos(yl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  -(-1/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.sin(zl)*np.cos(yl)*np.log(3)-1/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))*np.sin(yl)*np.log(3))*u0/(3**a6)/wph+(1/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.sin(zl)*np.cos(yl)+1/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))*np.sin(yl))*u0/(3**a6)/wph*np.log(3)+2/(3**a6)**2/wph**2*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.cos(zl)*v0*np.log(3)-1/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.sin(zl)*np.sin(yl)*np.log(3)+1/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))*np.cos(yl)*np.log(3)], \
#        [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            -(-1/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.sin(zl)*np.sin(yl)+1/(3**a6)/wph*np.cos(xr)*np.cos(zr)*np.cos(yl))*u0/(3**a6)/wph+1/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.sin(zl)*np.cos(yl)+1/(3**a6)/wph*np.cos(xr)*np.cos(zr)*np.sin(yl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        -1/(3**a6)**2/wph**2*np.sin(xr)*np.cos(zr)*np.cos(zl)*np.cos(yl)*u0+1/(3**a6)**2/wph**2*np.sin(xr)*np.cos(zr)*np.sin(zl)*v0+1/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.cos(zl)*np.sin(yl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     -(1/(3**a6)/wph*np.cos(xr)*np.cos(zr)*np.sin(zl)*np.cos(yl)-1/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.sin(yl))*u0/(3**a6)/wph-1/(3**a6)**2/wph**2*np.cos(xr)*np.cos(zr)*np.cos(zl)*v0+1/(3**a6)/wph*np.cos(xr)*np.cos(zr)*np.sin(zl)*np.sin(yl)+1/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.cos(yl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    -(-1/(3**a6)/wph*np.sin(xr)*np.sin(zr)*np.sin(zl)*np.cos(yl)-1/(3**a6)/wph*np.cos(xr)*np.sin(zr)*np.sin(yl))*u0/(3**a6)/wph+1/(3**a6)**2/wph**2*np.sin(xr)*np.sin(zr)*np.cos(zl)*v0-1/(3**a6)/wph*np.sin(xr)*np.sin(zr)*np.sin(zl)*np.sin(yl)+1/(3**a6)/wph*np.cos(xr)*np.sin(zr)*np.cos(yl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        -(-1/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.sin(zl)*np.cos(yl)*np.log(3)-1/(3**a6)/wph*np.cos(xr)*np.cos(zr)*np.sin(yl)*np.log(3))*u0/(3**a6)/wph+(1/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.sin(zl)*np.cos(yl)+1/(3**a6)/wph*np.cos(xr)*np.cos(zr)*np.sin(yl))*u0/(3**a6)/wph*np.log(3)+2/(3**a6)**2/wph**2*np.sin(xr)*np.cos(zr)*np.cos(zl)*v0*np.log(3)-1/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.sin(zl)*np.sin(yl)*np.log(3)+1/(3**a6)/wph*np.cos(xr)*np.cos(zr)*np.cos(yl)*np.log(3)], \
#        [                                                                                                                                                                                                                                                                                                                                                                                                      -(-(-u0/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)+np.sin(xr)*np.sin(zr)*np.sin(yr)+np.cos(xr)*np.cos(yr))*np.sin(zl)*np.sin(yl)-(u0/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))+1/(3**a6)/wph*v0*np.cos(xr)*np.cos(zr)-np.cos(xr)*np.sin(zr)*np.sin(yr)+np.sin(xr)*np.cos(yr))*np.cos(yl))*u0/(3**a6)/wph+(-u0/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)+np.sin(xr)*np.sin(zr)*np.sin(yr)+np.cos(xr)*np.cos(yr))*np.sin(zl)*np.cos(yl)-(u0/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))+1/(3**a6)/wph*v0*np.cos(xr)*np.cos(zr)-np.cos(xr)*np.sin(zr)*np.sin(yr)+np.sin(xr)*np.cos(yr))*np.sin(yl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        -(-u0/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)+np.sin(xr)*np.sin(zr)*np.sin(yr)+np.cos(xr)*np.cos(yr))*np.cos(zl)*np.cos(yl)*u0/(3**a6)/wph+(-u0/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)+np.sin(xr)*np.sin(zr)*np.sin(yr)+np.cos(xr)*np.cos(yr))*np.sin(zl)/(3**a6)/wph*v0+(-u0/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)+np.sin(xr)*np.sin(zr)*np.sin(yr)+np.cos(xr)*np.cos(yr))*np.cos(zl)*np.sin(yl),                                                                                                                                                                                                                                                                                                                                                -((-u0/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.cos(xr)*np.cos(zr)+np.cos(xr)*np.sin(zr)*np.sin(yr)-np.sin(xr)*np.cos(yr))*np.sin(zl)*np.cos(yl)-(u0/(3**a6)/wph*(-np.sin(xr)*np.sin(zr)*np.cos(yr)+np.cos(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)+np.sin(xr)*np.sin(zr)*np.sin(yr)+np.cos(xr)*np.cos(yr))*np.sin(yl))*u0/(3**a6)/wph-(-u0/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.cos(xr)*np.cos(zr)+np.cos(xr)*np.sin(zr)*np.sin(yr)-np.sin(xr)*np.cos(yr))*np.cos(zl)/(3**a6)/wph*v0+(-u0/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.cos(xr)*np.cos(zr)+np.cos(xr)*np.sin(zr)*np.sin(yr)-np.sin(xr)*np.cos(yr))*np.sin(zl)*np.sin(yl)+(u0/(3**a6)/wph*(-np.sin(xr)*np.sin(zr)*np.cos(yr)+np.cos(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)+np.sin(xr)*np.sin(zr)*np.sin(yr)+np.cos(xr)*np.cos(yr))*np.cos(yl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             -((-u0/(3**a6)/wph*(-np.sin(xr)*np.sin(zr)*np.sin(yr)-np.cos(xr)*np.cos(yr))+np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.sin(zl)*np.cos(yl)-(u0/(3**a6)/wph*(-np.cos(xr)*np.sin(zr)*np.sin(yr)+np.sin(xr)*np.cos(yr))-np.cos(xr)*np.sin(zr)*np.cos(yr)-np.sin(xr)*np.sin(yr))*np.sin(yl))*u0/(3**a6)/wph-(-u0/(3**a6)/wph*(-np.sin(xr)*np.sin(zr)*np.sin(yr)-np.cos(xr)*np.cos(yr))+np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.cos(zl)/(3**a6)/wph*v0+(-u0/(3**a6)/wph*(-np.sin(xr)*np.sin(zr)*np.sin(yr)-np.cos(xr)*np.cos(yr))+np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.sin(zl)*np.sin(yl)+(u0/(3**a6)/wph*(-np.cos(xr)*np.sin(zr)*np.sin(yr)+np.sin(xr)*np.cos(yr))-np.cos(xr)*np.sin(zr)*np.cos(yr)-np.sin(xr)*np.sin(yr))*np.cos(yl),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            -((-u0/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.cos(yr)+1/(3**a6)/wph*v0*np.sin(xr)*np.sin(zr)+np.sin(xr)*np.cos(zr)*np.sin(yr))*np.sin(zl)*np.cos(yl)-(u0/(3**a6)/wph*np.cos(xr)*np.cos(zr)*np.cos(yr)-1/(3**a6)/wph*v0*np.cos(xr)*np.sin(zr)-np.cos(xr)*np.cos(zr)*np.sin(yr))*np.sin(yl))*u0/(3**a6)/wph-(-u0/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.cos(yr)+1/(3**a6)/wph*v0*np.sin(xr)*np.sin(zr)+np.sin(xr)*np.cos(zr)*np.sin(yr))*np.cos(zl)/(3**a6)/wph*v0+(-u0/(3**a6)/wph*np.sin(xr)*np.cos(zr)*np.cos(yr)+1/(3**a6)/wph*v0*np.sin(xr)*np.sin(zr)+np.sin(xr)*np.cos(zr)*np.sin(yr))*np.sin(zl)*np.sin(yl)+(u0/(3**a6)/wph*np.cos(xr)*np.cos(zr)*np.cos(yr)-1/(3**a6)/wph*v0*np.cos(xr)*np.sin(zr)-np.cos(xr)*np.cos(zr)*np.sin(yr))*np.cos(yl), -((u0/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.log(3)+1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)*np.log(3))*np.sin(zl)*np.cos(yl)-(-u0/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))*np.log(3)-1/(3**a6)/wph*v0*np.cos(xr)*np.cos(zr)*np.log(3))*np.sin(yl))*u0/(3**a6)/wph+((-u0/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)+np.sin(xr)*np.sin(zr)*np.sin(yr)+np.cos(xr)*np.cos(yr))*np.sin(zl)*np.cos(yl)-(u0/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))+1/(3**a6)/wph*v0*np.cos(xr)*np.cos(zr)-np.cos(xr)*np.sin(zr)*np.sin(yr)+np.sin(xr)*np.cos(yr))*np.sin(yl))*u0/(3**a6)/wph*np.log(3)-(u0/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.log(3)+1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)*np.log(3))*np.cos(zl)/(3**a6)/wph*v0+(-u0/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))-1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)+np.sin(xr)*np.sin(zr)*np.sin(yr)+np.cos(xr)*np.cos(yr))*np.cos(zl)/(3**a6)/wph*v0*np.log(3)+(u0/(3**a6)/wph*(np.sin(xr)*np.sin(zr)*np.cos(yr)-np.cos(xr)*np.sin(yr))*np.log(3)+1/(3**a6)/wph*v0*np.sin(xr)*np.cos(zr)*np.log(3))*np.sin(zl)*np.sin(yl)+(-u0/(3**a6)/wph*(np.cos(xr)*np.sin(zr)*np.cos(yr)+np.sin(xr)*np.sin(yr))*np.log(3)-1/(3**a6)/wph*v0*np.cos(xr)*np.cos(zr)*np.log(3))*np.cos(yl)]]
#
#
#        ds = \
#        [   2*((m21*f1+m22*f2+f3)*m11+(m21*f4+m22*f5+f6)*m12+m21*f7+m22*f8+f9)/((-f2*m11-f5*m12-f8)**2+(f1*m11+f4*m12+f7)**2+(m21*f4+m22*f5+f6)**2+(-m21*f1-m22*f2-f3)**2)*m21*m11-((m21*f1+m22*f2+f3)*m11+(m21*f4+m22*f5+f6)*m12+m21*f7+m22*f8+f9)**2/((-f2*m11-f5*m12-f8)**2+(f1*m11+f4*m12+f7)**2+(m21*f4+m22*f5+f6)**2+(-m21*f1-m22*f2-f3)**2)**2*(2*(f1*m11+f4*m12+f7)*m11-2*(-m21*f1-m22*f2-f3)*m21), 2*((m21*f1+m22*f2+f3)*m11+(m21*f4+m22*f5+f6)*m12+m21*f7+m22*f8+f9)/((-f2*m11-f5*m12-f8)**2+(f1*m11+f4*m12+f7)**2+(m21*f4+m22*f5+f6)**2+(-m21*f1-m22*f2-f3)**2)*m22*m11-((m21*f1+m22*f2+f3)*m11+(m21*f4+m22*f5+f6)*m12+m21*f7+m22*f8+f9)**2/((-f2*m11-f5*m12-f8)**2+(f1*m11+f4*m12+f7)**2+(m21*f4+m22*f5+f6)**2+(-m21*f1-m22*f2-f3)**2)**2*(-2*(-f2*m11-f5*m12-f8)*m11-2*(-m21*f1-m22*f2-f3)*m22),                                   2*((m21*f1+m22*f2+f3)*m11+(m21*f4+m22*f5+f6)*m12+m21*f7+m22*f8+f9)/((-f2*m11-f5*m12-f8)**2+(f1*m11+f4*m12+f7)**2+(m21*f4+m22*f5+f6)**2+(-m21*f1-m22*f2-f3)**2)*m11-((m21*f1+m22*f2+f3)*m11+(m21*f4+m22*f5+f6)*m12+m21*f7+m22*f8+f9)**2/((-f2*m11-f5*m12-f8)**2+(f1*m11+f4*m12+f7)**2+(m21*f4+m22*f5+f6)**2+(-m21*f1-m22*f2-f3)**2)**2*(2*m21*f1+2*m22*f2+2*f3),    2*((m21*f1+m22*f2+f3)*m11+(m21*f4+m22*f5+f6)*m12+m21*f7+m22*f8+f9)/((-f2*m11-f5*m12-f8)**2+(f1*m11+f4*m12+f7)**2+(m21*f4+m22*f5+f6)**2+(-m21*f1-m22*f2-f3)**2)*m21*m12-((m21*f1+m22*f2+f3)*m11+(m21*f4+m22*f5+f6)*m12+m21*f7+m22*f8+f9)**2/((-f2*m11-f5*m12-f8)**2+(f1*m11+f4*m12+f7)**2+(m21*f4+m22*f5+f6)**2+(-m21*f1-m22*f2-f3)**2)**2*(2*(f1*m11+f4*m12+f7)*m12+2*(m21*f4+m22*f5+f6)*m21),  2*((m21*f1+m22*f2+f3)*m11+(m21*f4+m22*f5+f6)*m12+m21*f7+m22*f8+f9)/((-f2*m11-f5*m12-f8)**2+(f1*m11+f4*m12+f7)**2+(m21*f4+m22*f5+f6)**2+(-m21*f1-m22*f2-f3)**2)*m22*m12-((m21*f1+m22*f2+f3)*m11+(m21*f4+m22*f5+f6)*m12+m21*f7+m22*f8+f9)**2/((-f2*m11-f5*m12-f8)**2+(f1*m11+f4*m12+f7)**2+(m21*f4+m22*f5+f6)**2+(-m21*f1-m22*f2-f3)**2)**2*(-2*(-f2*m11-f5*m12-f8)*m12+2*(m21*f4+m22*f5+f6)*m22),                                   2*((m21*f1+m22*f2+f3)*m11+(m21*f4+m22*f5+f6)*m12+m21*f7+m22*f8+f9)/((-f2*m11-f5*m12-f8)**2+(f1*m11+f4*m12+f7)**2+(m21*f4+m22*f5+f6)**2+(-m21*f1-m22*f2-f3)**2)*m12-((m21*f1+m22*f2+f3)*m11+(m21*f4+m22*f5+f6)*m12+m21*f7+m22*f8+f9)**2/((-f2*m11-f5*m12-f8)**2+(f1*m11+f4*m12+f7)**2+(m21*f4+m22*f5+f6)**2+(-m21*f1-m22*f2-f3)**2)**2*(2*m21*f4+2*m22*f5+2*f6),                                   2*((m21*f1+m22*f2+f3)*m11+(m21*f4+m22*f5+f6)*m12+m21*f7+m22*f8+f9)/((-f2*m11-f5*m12-f8)**2+(f1*m11+f4*m12+f7)**2+(m21*f4+m22*f5+f6)**2+(-m21*f1-m22*f2-f3)**2)*m21-((m21*f1+m22*f2+f3)*m11+(m21*f4+m22*f5+f6)*m12+m21*f7+m22*f8+f9)**2/((-f2*m11-f5*m12-f8)**2+(f1*m11+f4*m12+f7)**2+(m21*f4+m22*f5+f6)**2+(-m21*f1-m22*f2-f3)**2)**2*(2*f1*m11+2*f4*m12+2*f7),                                   2*((m21*f1+m22*f2+f3)*m11+(m21*f4+m22*f5+f6)*m12+m21*f7+m22*f8+f9)/((-f2*m11-f5*m12-f8)**2+(f1*m11+f4*m12+f7)**2+(m21*f4+m22*f5+f6)**2+(-m21*f1-m22*f2-f3)**2)*m22-((m21*f1+m22*f2+f3)*m11+(m21*f4+m22*f5+f6)*m12+m21*f7+m22*f8+f9)**2/((-f2*m11-f5*m12-f8)**2+(f1*m11+f4*m12+f7)**2+(m21*f4+m22*f5+f6)**2+(-m21*f1-m22*f2-f3)**2)**2*(2*f2*m11+2*f5*m12+2*f8),                                                                                                                                                                                                                             2*((m21*f1+m22*f2+f3)*m11+(m21*f4+m22*f5+f6)*m12+m21*f7+m22*f8+f9)/((-f2*m11-f5*m12-f8)**2+(f1*m11+f4*m12+f7)**2+(m21*f4+m22*f5+f6)**2+(-m21*f1-m22*f2-f3)**2)]
#
#        jac_temp = np.float64(ds).dot(np.float64(df))
#        jac.append(jac_temp)
#    return err, jac


def quasi_euclidean_rectification(stereoPairs, epiSets, focal, fixedFocal):
    """
    function that starts the routine for a list of n stereoPairs

    inputs: stereoPairs    - list of n stereo images
            epiSets        - data containing the epipolar geometry
                              list of list of dictionaries
            angleCheck     - in case of set to true the angles of the image
                              are checked to lie in 90+-1°, otherwise the next
                              set of point correspondences is used to find
                              rectyfing transformations with less distortions
            focal         - value to compute the actual focal length,
                              empircally determined value accoridng to Fusiello
                              f = focal*(w+h)
            fixedFocal     - if set True the focal length is not determined
                              by the non-linear estimation and set fixed as
                              an argument (it is actually used in pixels
                              according to f = focal*(w+h) )
    """

    print("START UNCALIBRATED RECTIFICATION:", end="", flush=True)

    rectData = []
    i = 0

    for pair in stereoPairs:
        print("\n Rectify images of StereoPair {}: ".format(
            i + 1), end="", flush=True)

        img1 = pair[0]
        img2 = pair[1]

        (img1R,
         img2R,
         H1,
         H2,
         Ht,
         angles,
         corners1,
         corners2,
         uvMinMax,
         epi_rect,
         ) = qer_rectification_bb(img1, img2, epiSets, i, focal, fixedFocal)

        rect_datum = {}
        rect_datum["img1Rect"] = img1R
        rect_datum["img2Rect"] = img2R
        rect_datum["H1"] = H1
        rect_datum["H2"] = H2
        rect_datum["Ht"] = Ht
        rect_datum["angles"] = angles
        rect_datum["corners1"] = corners1
        rect_datum["corners2"] = corners2
        rect_datum["uvMinMax"] = uvMinMax
        rect_datum["epi"] = epi_rect

        rectData.append(rect_datum)
        i += 1

    return rectData


def qer_rectification_bb(img1, img2, epiSets, i, focal, fixedFocal):
    """
    perform the uncalibrated epipolar rectification according to
    fusiello's paper 'Quasi-Euclidean Epipolar Rectification' and returns
    the rectified images in the smallest bounding box of both images
    inputs: img1, img2      - images to be rectified
            epiSets        - data containing the epipolar geometry
                              list of list of dictionaries
            angleCheck     - in case of set to true the angles of the image
                              are checked to lie in 90+-1°, otherwise the next
                              set of point correspondences is used to find
                              rectyfing transformations with less distortions
            focal         - value to compute the actual focal length,
                              empircally determined value accoridng to Fusiello
                              f = focal*(w+h)
            fixedFocal     - if set True the focal length is not determined
                              by the non-linear estimation and set fixed as
                              an argument (it is actually used in pixels
                              according to f = focal*(w+h) )
    """
    # select points and fundamental matrix of the first set
    mr = epiSets[i]["pts1"]
    ml = epiSets[i]["pts2"]
    F = epiSets[i]["F"]

    # get shapes of the images
    h1, w1 = np.shape(img2)
    h2, w2 = np.shape(img1)

    if w1 != w2 or h1 != h2:
        print(
            "The images don't have the same dimensions - no rectified images returned."
        )
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    else:
        w = w1
        h = h1

    if fixedFocal == True:
        print("Using a fixed focal length..", end="", flush=True)
        paramsStartFixed = np.array([0, 0, 0, 0, 0])  # -0.5875
        paramBoundsFixed = (
            [-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2],
            [np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2],
        )
        result = least_squares(
            qer_cost_fixed_focal,
            paramsStartFixed,
            method="trf",
            bounds=paramBoundsFixed,
            max_nfev=1000,
            ftol=1e-8,
            xtol=1e-8,
            args=(w, h, ml, mr, focal),
        )
        af = result.x
    else:
        # start vector for the angles of the transformation matrices and the focal length
        paramsStart = np.array([0, 0, 0, 0, 0, 0])
        # bounds of the angles and the focal length, the focal length gets recomputed obtaining the
        # minimized value
        paramBounds = ([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, -1.6],
                       [np.pi / 2, np.pi / 2, np.pi / 2,
                           np.pi / 2, np.pi / 2, 1.6],
                       )  # -1.1
        # trusted region reflective method to minimize the sampsons error getting the parameteres
        # of the rectfying transformations using special constraints ("compare uncalibrated
        # epipolar rectification, fusiello")
        result = least_squares(qer_cost,
                               paramsStart,
                               method="trf",
                               bounds=paramBounds,
                               max_nfev=1000,
                               ftol=1e-8,
                               xtol=1e-8,
                               args=(w, h, ml, mr, focal),
                               )
        af = result.x

    yl = af[0]
    zl = af[1]
    xr = af[2]
    yr = af[3]
    zr = af[4]

    if len(af) == 6:
        f = focal ** af[5] * (w + h)
    else:
        f = focal * (w + h)

    # define the old intrinsics
    Kol = np.float64([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    Kor = np.float64([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])

    # get the rotation matrices
    Rl = euler_R([0, yl, zl])
    Rr = euler_R([xr, yr, zr])

    # new intrinsics: arbitrary chosen
    Knl = (Kol + Kor) / 2
    Knr = Knl

    # compute rectifying collineations
    Hr = Knr.dot(Rr).dot(inv(Kor))
    Hl = Knl.dot(Rl).dot(inv(Kol))

    # centering LEFT image
    px = Hl.dot(np.float64([w / 2, h / 2, 1]))
    dl = [w / 2, h / 2] - np.float64([px[0] / px[2], px[1] / px[2]])

    # centering RIGHT image
    px2 = Hr.dot(np.float64([w / 2, h / 2, 1]))
    dr = [w / 2, h / 2] - np.float64([px2[0] / px2[2], px2[1] / px2[2]])

    # vertical diplacement must be the same
    c1 = Knl[0, 2] + dl[0]
    c2 = Knl[1, 2] + dl[1]
    c3 = Knr[0, 2] + dr[0]
    c4 = Knr[1, 2] + dl[1]

    # modify new intrinsic
    KnlC = np.float64([[Knl[0, 0], Knl[0, 1], c1],
                       [Knl[1, 0], Knl[1, 1], c2],
                       [Knl[2, 0], Knl[2, 1], Knl[2, 2]],
                       ]
                      )

    KnrC = np.float64([[Knr[0, 0], Knr[0, 1], c3],
                       [Knr[1, 0], Knr[1, 1], c4],
                       [Knr[2, 0], Knr[2, 1], Knr[2, 2]],
                       ]
                      )

    # re-compute collineations with centering
    Hr = np.float32(KnrC.dot(Rr).dot(inv(Kor)))
    Hl = np.float32(KnlC.dot(Rl).dot(inv(Kol)))

    # just to ensure corners are correctly compute
    corners = np.float32([[0, 0], [0, h], [w, 0], [w, h]]).reshape(-1, 1, 2)

    # transformed corners of the rectified image
    # equivalent to usage of openCV's perspectiveTransform function
    # cornersTl = np.float32(np.transpose(transform_2D(corners.T, Hl)))
    # cornersTr = np.float32(np.transpose(transform_2D(corners.T, Hr)))
    cornersTl = cv.perspectiveTransform(corners, Hl)
    cornersTr = cv.perspectiveTransform(corners, Hr)

    uMin = int(np.floor(min(cornersTl[0, :, 0],
                            cornersTl[1, :, 0],
                            cornersTr[0, :, 0],
                            cornersTr[1, :, 0],
                            )
                        )
               )  # Spaltenanzahl - min_wl

    uMax = int(np.ceil(max(cornersTl[2, :, 0],
                           cornersTl[3, :, 0],
                           cornersTr[2, :, 0],
                           cornersTr[3, :, 0],
                           )
                       )
               )  # max_w

    vMin = int(np.floor(min(cornersTl[0, :, 1],
                            cornersTl[2, :, 1],
                            cornersTr[0, :, 1],
                            cornersTr[2, :, 1],
                            )
                        )
               )  # Zeilenanzahl - min_hl

    vMax = int(np.ceil(max(cornersTl[1, :, 1],
                           cornersTl[3, :, 1],
                           cornersTr[1, :, 1],
                           cornersTr[3, :, 1],
                           )
                       )
               )  # max_hl

    # translation is needed to transform the new image into its
    # bounding box
    t = [-uMin, -vMin]
    Ht = np.float32([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    # compute the angles of the image corners
    angles = compute_image_corner_angles(cornersTl, cornersTr)

    img2 = np.float32(img2)
    img1 = np.float32(img1)

    # apply the rectifying transformations to the images
    img2R = cv.warpPerspective(img2, Ht.dot(Hl), (uMax - uMin, vMax - vMin))
    img1R = cv.warpPerspective(img1, Ht.dot(Hr), (uMax - uMin, vMax - vMin))

    print("OK", end="", flush=True)

    return (img1R,
            img2R,
            Hr,
            Hl,
            Ht,
            angles,
            cornersTl,
            cornersTr,
            (uMin, uMax, vMin, vMax),
            (mr, ml, F),
            )


def get_angle(v1, v2):
    """
    get the angle between two vectors
    """
    if np.shape(v1)[0] == 1 or np.shape(v1)[0] == 1:
        sys.exit("get_angle() needs an input in form of nx1")
    angle = np.arccos((np.dot(v1.T, v2)) / (la.norm(v1) * la.norm(v2)))
    return np.rad2deg(angle)


def compute_image_corner_angles(cornersTl, cornersTr):
    # vectors between the image corners
    side12l = (cornersTl[1] - cornersTl[0]).T
    side13l = (cornersTl[2] - cornersTl[0]).T
    side21l = (cornersTl[0] - cornersTl[1]).T
    side24l = (cornersTl[3] - cornersTl[1]).T
    side31l = (cornersTl[0] - cornersTl[2]).T
    side34l = (cornersTl[3] - cornersTl[2]).T
    side42l = (cornersTl[1] - cornersTl[3]).T
    side43l = (cornersTl[2] - cornersTl[3]).T

    # angles of the image corners
    phiL1 = get_angle(side12l, side13l)
    phiL2 = get_angle(side21l, side24l)
    phiL3 = get_angle(side31l, side34l)
    phiL4 = get_angle(side42l, side43l)
    anglesImgL = [phiL1, phiL2, phiL3, phiL4]

    # vectors between the image corners
    side12r = (cornersTr[1] - cornersTr[0]).T
    side13r = (cornersTr[2] - cornersTr[0]).T
    side21r = (cornersTr[0] - cornersTr[1]).T
    side24r = (cornersTr[3] - cornersTr[1]).T
    side31r = (cornersTr[0] - cornersTr[2]).T
    side34r = (cornersTr[3] - cornersTr[2]).T
    side42r = (cornersTr[1] - cornersTr[3]).T
    side43r = (cornersTr[2] - cornersTr[3]).T

    # angles of the image corners
    phiR1 = get_angle(side12r, side13r)
    phiR2 = get_angle(side21r, side24r)
    phiR3 = get_angle(side31r, side34r)
    phiR4 = get_angle(side42r, side43r)
    anglesImgR = [phiR1, phiR2, phiR3, phiR4]

    # check angles of the rectified images
    angles = np.squeeze(np.float64([anglesImgL, anglesImgR]))
    return angles


###############################################################################
#   QUASI EUCLIDEAN RECTIFICATION FUNCTIONS - END
###############################################################################

###############################################################################
#    AFFINE RECTIFICATION FUNCTIONS - START
###############################################################################


def rectifying_rigid_transformations(F, pts1, pts2):
    """
    Computes two rigid transformations from an affine fundamental matrix and
    its related point correspondences.

    inputs:
        F       - 3x3 numpy array, the fundamental matrix
        pts1    - 3xn numpy array
        pts2    - 3xn numpy array

    outputs:
        S1, S2  - two rigid transformations to rectify the two stereo images
    """

    # check that the input matrix is an affine fundamental matrix
    assert np.shape(F) == (3, 3)
    assert np.linalg.matrix_rank(F) == 2
    assert F[0, 0] == 0
    assert F[0, 1] == 0
    assert F[1, 0] == 0
    assert F[1, 1] == 0

    # notations, same conventions as in H&Z
    c = F[2, 0]
    d = F[2, 1]
    a = F[0, 2]
    b = F[1, 2]
    # computing the translation by the transformed point correspondences, e
    # is actually not needed
    e = F[2, 2]
    if e < 0:
        F = -F

    # rotations
    s = np.sqrt(a * a + b * b)
    r = np.sqrt(c * c + d * d)
    if a < 0 and b < 0:
        R1 = (1.0 / r) * np.array([[d, -c], [c, d]])
        R2 = (1.0 / s) * np.array([[-b, a], [-a, -b]])
    elif c < 0 and d < 0:
        R1 = (1.0 / r) * np.array([[-d, c], [-c, -d]])
        R2 = (1.0 / s) * np.array([[b, -a], [a, b]])
    elif a < 0 and d < 0:
        R1 = (1.0 / r) * np.array([[-d, c], [-c, -d]])
        R2 = (1.0 / s) * np.array([[b, -a], [a, b]])
    elif b < 0 and c < 0:
        R1 = (1.0 / r) * np.array([[d, -c], [c, d]])
        R2 = (1.0 / s) * np.array([[-b, a], [-a, -b]])
    else:
        raise Exception("something is wrong with the rotation matrices sign")

    # rectifying transformation/rotation matrices without translation
    T1 = np.zeros((3, 3))
    T1[0:2, 0:2] = R1
    T1[1, 2] = 0
    T1[2, 2] = 1

    T2 = np.zeros((3, 3))
    T2[0:2, 0:2] = R2
    T2[1, 2] = 0
    T2[2, 2] = 1

    # determine the necessary translation to align the epipolar lines
    pts_t1 = T1.dot(pts1)
    pts_t2 = T2.dot(pts2)
    translation = pts_t1[1, :] - pts_t2[1, :]
    t = np.mean(translation)

    # rigid transformation matrices containing rotations and translations
    S1 = np.zeros((3, 3))
    S1[0:2, 0:2] = R1
    S1[1, 2] = 0
    S1[2, 2] = 1

    S2 = np.zeros((3, 3))
    S2[0:2, 0:2] = R2
    S2[1, 2] = t
    S2[2, 2] = 1

    return S1, S2


def rectifying_translation(pts1, pts2):
    """
    Computes an aligning translation from point correspondences.

    inputs:

        pts1   - 3xn numpy array
        pts2   - 3xn numpy array

    outputs:
        S1, S2 - two transformation matrices that contain the translations
    """

    R1 = np.eye(2)
    R2 = R1
    # rectifying transformation/rotation matrices without translation
    T1 = np.zeros((3, 3))
    T1[0:2, 0:2] = R1
    T1[1, 2] = 0
    T1[2, 2] = 1

    T2 = np.zeros((3, 3))
    T2[0:2, 0:2] = R2
    T2[1, 2] = 0
    T2[2, 2] = 1

    # determine the necessary translation to align the epipolar lines
    pts_t1 = T1.dot(pts1)
    pts_t2 = T2.dot(pts2)
    translation = pts_t1[1, :] - pts_t2[1, :]
    t = np.mean(translation)

    # rigid transformation matrices containing rotations and translations
    S1 = np.zeros((3, 3))
    S1[0:2, 0:2] = R1
    S1[1, 2] = 0
    S1[2, 2] = 1

    S2 = np.zeros((3, 3))
    S2[0:2, 0:2] = R2
    S2[1, 2] = t
    S2[2, 2] = 1

    return S1, S2


def get_angle_from_cos_and_sin(c, s):
    if s >= 0:
        return np.arccos(c)
    else:
        return -np.arccos(c)


def affine_rectification(F, pts1, pts2):
    """
    computes two affine rectifying transformations from an affine
    fundamental matrix and its point correspondences according to Hartley

    inputs:
         F       - 3x3 numpy array, the fundamental matrix

    outputs:
        Ta1, Ta2 - two affine transformations to perform the stereo rectification
    """
    # check that the input matrix is an affine fundamental matrix
    assert np.shape(F) == (3, 3)
    assert np.linalg.matrix_rank(F) == 2
    assert F[0, 0] == 0
    assert F[0, 1] == 0
    assert F[1, 0] == 0
    assert F[1, 1] == 0

    # notations, same conventions as in H&Z
    c = F[2, 0]
    d = F[2, 1]
    a = F[0, 2]
    b = F[1, 2]

    s = np.sqrt(a * a + b * b)
    r = np.sqrt(c * c + d * d)
    if a < 0 and b < 0:
        R1 = (1.0 / r) * np.array([[d, -c], [c, d]])
        R2 = (1.0 / s) * np.array([[-b, a], [-a, -b]])
    elif c < 0 and d < 0:
        R1 = (1.0 / r) * np.array([[-d, c], [-c, -d]])
        R2 = (1.0 / s) * np.array([[b, -a], [a, b]])
    elif a < 0 and d < 0:
        R1 = (1.0 / r) * np.array([[-d, c], [-c, -d]])
        R2 = (1.0 / s) * np.array([[b, -a], [a, b]])
    elif b < 0 and c < 0:
        R1 = (1.0 / r) * np.array([[d, -c], [c, d]])
        R2 = (1.0 / s) * np.array([[-b, a], [-a, -b]])
    else:
        raise Exception("something is wrong with the rotation matrices sign")

    # rectifying transformation/rotation matrices without translation
    T1 = np.zeros((3, 3))
    T1[0:2, 0:2] = R1
    T1[1, 2] = 0
    T1[2, 2] = 1

    T2 = np.zeros((3, 3))
    T2[0:2, 0:2] = R2
    T2[1, 2] = 0
    T2[2, 2] = 1

    # rotate points
    pts1R = T1.dot(pts1.T)
    pts2R = T2.dot(pts2.T)

    # align and scale the y-axis
    Aone = np.ones((np.shape(pts1R)[1]))
    # y-coordinates and ones image 1
    A = np.vstack((pts1R[1, :], Aone))
    # y-coordinates image 2
    b = pts2R[1, :]
    # this formulation results in the equation system Ax = b = a*y1,i + ty = y2
    # eigentlich ist es [y_1,i ,  1 ]*x = b = y2
    # so it is solved for y-scaling in image 1 and y-translation for the
    # point correspondences in image 1; ys = scale, y0 = translation
    (ys, y0), res, rank, s = np.linalg.lstsq(A.T, b, rcond=None)
    # get the aligned and scaled pixels
    pts1RSt = np.array([[1, 0, 0], [0, ys, y0], [0, 0, 1]]).dot(pts1R)

    # next step:
    # minimize x-deviation to reduce the search range of the matching-algorithm
    # same way of computation: Ax = b => [x1,i y1,i 1] * x = [x2,i]
    # yields the equation system: xs*x1,i + y1,i*xsk + x0 = x2,i

    # scaled and aligned points of image one
    Ax = pts1RSt
    # x-coordinates of image 2
    bx = pts2R[0, :]
    # solve for x-scale - xs, skew, xsk and translation x0
    (xs, xsk, x0), res, rank, s = np.linalg.lstsq(Ax.T, bx, rcond=None)

    # compute the final point corrdinates of the correspondences
    # pts1_rec = np.array([[xs, xsk/2, x0], [0, 1, 0], [0, 0, 1]]).dot(pts1RSt)
    # pts2_rec = np.array([[1, -xsk/2, 0], [0, 1, 0], [0, 0, 1]]).dot(pts2R)

    # get the final affine rectifying transformations
    Ta1 = np.array([[xs, xsk / 2, x0], [0, ys, y0], [0, 0, 1]]).dot(T1)
    Ta2 = np.array([[1, -xsk / 2, 0], [0, 1, 0], [0, 0, 1]]).dot(T2)
    return Ta1, Ta2


def rectifying_similarities(F):
    """
    computes two similarity transformations from an affine fundamental matrix
    based on the findings of
    Shimshoni "A geometric interpretation of weak-perspective motion"

    inputs:
        F       - 3x3 numpy array, the fundamental matrix

    outputs:
        S1, S2  - two similarity transformations to perform the
                  stereo rectification consistent with Fa
    """
    # check that the input matrix is an affine fundamental matrix
    assert np.shape(F) == (3, 3)
    assert np.linalg.matrix_rank(F) == 2
    assert F[0, 0] == 0
    assert F[0, 1] == 0
    assert F[1, 0] == 0
    assert F[1, 1] == 0

    # notations, THE SAME CONVENTION as in H&Z
    c = F[2, 0]
    d = F[2, 1]
    a = F[0, 2]
    b = F[1, 2]
    e = F[2, 2]

    # rotations
    r = np.sqrt(a * a + b * b)
    s = np.sqrt(c * c + d * d)
    R2 = (1.0 / r) * np.array([[-b, a], [-a, -b]])
    R1 = (1.0 / s) * np.array([[d, -c], [c, d]])

    # zoom and translation
    z = s / r

    # compute translation from e and only for the image that is the reference
    # img 2 gets scaled to match img 1
    # (scaling is not divided onto both images and only applied to image 2)
    t = e / s

    # output similarities
    S1 = np.zeros((3, 3))
    S1[0:2, 0:2] = R1
    S1[1, 2] = 0
    S1[2, 2] = 1

    S2 = np.zeros((3, 3))
    S2[0:2, 0:2] = (1.0 / z) * R2
    S2[1, 2] = -t
    S2[2, 2] = 1

    if S2[0, 0] < 0:
        mir = np.array(np.mat("-1 -1 -1; -1 -1 -1; 1 1 1"))
        S1 = S1 * mir
        S2 = S2 * mir

    return S1, S2


###############################################################################
#    AFFINE RECTIFICATION FUNCTIONS - END
###############################################################################


def transform_2D(pts, H):
    """
    pts   - 3xn, homogeneous points x,y,1
    H     - 3x3
    pts_T - nx2
    """
    pts_T_un = H.dot(pts)
    pts_T = pts_T_un[0:2, :] / np.array((pts_T_un[2, :], pts_T_un[2, :]))
    return pts_T


def get_rect_error_from_H(Ha, Hb, pts1, pts2, F, name="say what"):
    """
    determine the epipolar distances after the rectification
    """
    # transform the points into the rectified image
    pts1_T = utils.homogenize(transform_2D(pts1.T, Ha)).T
    pts2_T = utils.homogenize(transform_2D(pts2.T, Hb)).T
    # create fundamental matrix for a rectified image
    F_r = np.matrix([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    # calculate the error after the rectification
    avg_err, max_err, error = dist.sym_epi_dist(pts1, pts2, F)
    avg_err_rect, max_err_rect, error_rect = dist.sym_epi_dist(
        pts1_T, pts2_T, F_r)
    print(name)
    print("avg error before rect:")
    print(avg_err)
    print("avg error after rect:")
    print(avg_err_rect)
    return error, error_rect


def get_rect_error(img,
                   img1,
                   H1,
                   H2,
                   Ht,
                   pts1,
                   pts2,
                   F,
                   title=0,
                   errorType="symepi",
                   plot="img",
                   scaleMax=1
                   ):
    """
    determine the epipolar distances after the rectification
    pts1, pts2 : nx3 array
    matrices like commonly usual
    H1, H2: 2D Rotation matrices
    Ht: centering transformation for bounding box computation
    """
    Ha = Ht.dot(H1)
    Hb = Ht.dot(H2)
    # transform the points into the rectified image
    pts1T = utils.homogenize(transform_2D(pts1.T, Ha)).T
    pts2T = utils.homogenize(transform_2D(pts2.T, Hb)).T
    # create fundamental matrix for a rectified image
    Fr = np.matrix([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
    # calculate the error after the rectification
    avg_err, max_err, error = dist.sym_epi_dist(pts1, pts2, F)
    avg_err_rect, max_err_rect, error_rect = dist.sym_epi_dist(
        pts1T, pts2T, Fr)
    # print tilte and error
    if title != 0:
        print("\n\n{}:".format(title))
    # just show 4 digits
    avgErr4 = int(avg_err * 10**4) / 10**4
    avgErrRect4 = int(avg_err_rect * 10**4) / 10**4
    print("    avg error - pre rect: {} post rec: {}".format(avgErr4, avgErrRect4))
    # if wanted plot the errors in the images
    if plot == "white":
        vis.printError(pts1T,
                       pts2T,
                       Fr,
                       name="{} - post rec".format(title),
                       display=1,
                       typ=errorType,
                       max_err=scaleMax,
                       )
        vis.printError(pts1,
                       pts2,
                       F,
                       name="{} - pre rec".format(title),
                       display=1,
                       typ=errorType,
                       max_err=scaleMax,
                       )
    elif plot == "img":
        vis.printErrorImage(img,
                            img1,
                            pts1T,
                            pts2T,
                            Fr,
                            name="{} - post rec".format(title),
                            display=1,
                            typ=errorType,
                            max_err=scaleMax,
                            rect=1
                            )
        vis.printErrorImage(img,
                            img1,
                            pts1,
                            pts2,
                            F,
                            name="{} - pre rec".format(title),
                            display=1,
                            typ=errorType,
                            max_err=scaleMax,
                            rect=0
                            )
    else:
        vis.printError(pts1T,
                       pts2T,
                       Fr,
                       name="{} - post rec".format(title),
                       display=0,
                       typ=errorType,
                       max_err=scaleMax
                       )
        vis.printError(pts1,
                       pts2,
                       F,
                       name="{} - pre rec".format(title),
                       display=0,
                       typ=errorType,
                       max_err=scaleMax
                       )
    return error, error_rect


def transform_image_bb(H, img):
    """
    transform an image an get the new bounding box
    """
    h, w = np.shape(img)
    corners = np.float32([[0, 0], [0, h], [w, 0], [w, h]]).reshape(-1, 1, 2)
    corners_T = cv.perspectiveTransform(corners, H)

    # Spaltenanzahl - min_wl
    uMin = np.floor(min(corners_T[0, :, 0], corners_T[1, :, 0]))

    # max_w
    uMax = np.ceil(max(corners_T[2, :, 0], corners_T[3, :, 0]))

    # Zeilenanzahl - min_hl
    vMin = np.floor(min(corners_T[0, :, 1], corners_T[2, :, 1]))

    # max_hl
    vMax = np.ceil(max(corners_T[1, :, 1], corners_T[3, :, 1]))

    t = [-uMin, -vMin]
    Ht = np.float32(np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]]))

    img_T = cv.warpPerspective(
        np.float32(img), np.float32(Ht.dot(H)), (uMax - uMin, vMax - vMin)
    )

    return img_T, Ht


def transform_two_images_bb(H1, H2, img1, img2):
    """
    transform two images and get the resulting bounding box for both images,
    images are equally centered in the process
    """

    h, w = np.shape(img1)
    h2, w2 = np.shape(img2)

    corners = np.float32([[0, 0], [0, h], [w, 0], [w, h]]).reshape(-1, 1, 2)
    corners2 = np.float32(
        [[0, 0], [0, h2], [w2, 0], [w2, h2]]).reshape(-1, 1, 2)

    cornersTl = cv.perspectiveTransform(corners, H1)
    cornersTr = cv.perspectiveTransform(corners2, H2)

    # Spaltenanzahl - min_wl
    uMin = int(np.floor(min(cornersTl[0, :, 0],
                            cornersTl[1, :, 0],
                            cornersTr[0, :, 0],
                            cornersTr[1, :, 0],
                            )
                        )
               )

    # max_w
    uMax = int(np.ceil(max(cornersTl[2, :, 0],
                           cornersTl[3, :, 0],
                           cornersTr[2, :, 0],
                           cornersTr[3, :, 0],
                           )
                       )
               )

    # Zeilenanzahl - min_hl
    vMin = int(np.floor(min(cornersTl[0, :, 1],
                            cornersTl[2, :, 1],
                            cornersTr[0, :, 1],
                            cornersTr[2, :, 1],
                            )
                        )
               )

    # max_hl
    vMax = int(np.ceil(max(cornersTl[1, :, 1],
                           cornersTl[3, :, 1],
                           cornersTr[1, :, 1],
                           cornersTr[3, :, 1],
                           )
                       )
               )

    t = [-uMin, -vMin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    img_1T = cv.warpPerspective(
        img1, np.float32(Ht.dot(H1)), (uMax - uMin, vMax - vMin), flags=cv.INTER_LINEAR
    )

    img_2T = cv.warpPerspective(
        img2, np.float32(Ht.dot(H2)), (uMax - uMin, vMax - vMin), flags=cv.INTER_LINEAR
    )

    return img_1T, img_2T, np.float32(Ht)
