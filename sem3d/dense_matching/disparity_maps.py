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
#                   dense matching of rectified stereo images and direct
#                   processing of the disparity map
#                   """


# libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
import numpy.linalg as la
from scipy import ndimage

# modules
import sem3d.utils.weighted_median_filter as wm


def sgbm(img1,
         img2,
         minDisp=-64,
         maxDisp=256,
         p1f=8,
         p2f=32,
         pfc=0,
         ur=5,
         dispDiff=1,
         sws=175,
         sr=3,
         windowSize=1,
         ):
    """
         minDisparity - Minimum possible disparity value. Normally, it is zero
                        but sometimes rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.
       numDisparities - Maximum disparity minus minimum disparity. The value is always greater than zero. In the current implementation, this parameter must be divisible by 16.
        SADWindowSize - Matched block size. It must be an odd number >=1. Normally, it should be somewhere in the 3..11 range.
                  P1  - The first parameter controlling the disparity smoothness. See below.
                  P2  - The second parameter controlling the disparity smoothness.
                        The larger the values are, the smoother the disparity is.
                        P1 is the penalty on the disparity change by plus or minus 1 between neighbor pixels.
                        P2 is the penalty on the disparity change by more than 1 between neighbor pixels.
                        The algorithm requires P2 > P1. See stereo_match.cpp sample where some reasonably good P1 and P2 values are shown
                        (like 8*number_of_image_channels*SADWindowSize*SADWindowSize and 32*number_of_image_channels*SADWindowSize*SADWindowSize, respectively).
        disp12MaxDiff - Maximum allowed difference (in integer pixel units) in the left-right disparity check.
                        Set it to a non-positive value to disable the check.
         preFilterCap - Truncation value for the prefiltered image pixels.
                        The algorithm first computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval.
                        The result values are passed to the Birchfield-Tomasi pixel cost function.
      uniquenessRatio - Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider
                        the found match correct. Normally, a value within the 5-15 range is good enough.
    speckleWindowSize - Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
                        Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
                        Maximum blob size of blobs that will be removed.
         speckleRange - Maximum disparity variation within each connected component.
                        If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
                        Normally, 1 or 2 is good enough.
                        Range of similar disparities within one blob (k*16).
               fullDP - Set it to true to run the full-scale two-pass dynamic programming algorithm.
                        It will consume O(W*H*numDisparities) bytes, which is large for 640x480 stereo and huge for HD-size pictures.
                        By default, it is set to false.
    """

    numDisp = (maxDisp) - minDisp
    stereo = cv2.StereoSGBM_create(minDisparity=minDisp,
                                   numDisparities=numDisp,
                                   blockSize=windowSize,
                                   preFilterCap=pfc,
                                   # 8*1*window_siobtainedze**2, 8*number_of_image_channels*SADWindowSize*SADWindowSize
                                   P1=p1f * (windowSize**2),
                                   # 32*1*window_size**2, 32*number_of_image_channels*SADWindowSize*SADWindowSize
                                   P2=p2f * (windowSize**2),
                                   disp12MaxDiff=dispDiff,
                                   uniquenessRatio=ur,
                                   speckleWindowSize=sws,  # 175
                                   speckleRange=sr,
                                   mode=cv2.STEREO_SGBM_MODE_HH,  # all 8 paths are used
                                   )
    dispSGBM = stereo.compute(img1, img2)
    return dispSGBM


def filter_disparity(dispMaps, method="bilateral", window=9, sigma=25.5):
    """
    Missing values need to be interpolated to apply most filters to the disparity map.
    inputs: disparity_maps      - list of 'number of stereo-pair images' entries
                                  (f.i. 5 for 5 stereo-pair images)
                                  each entry contains a disparity map
                                  in pixels (invalid pixels are minimum values)
    outputs: disp_filtered      - list of 'number of stereo-pair images' entries
                                  (f.i. 5 for 5 stereo-pair images)
                                  each entry contains a disparity map
                                  in pixels (invalid pixels are minimum values)
    """
    if len(dispMaps) > 100:
        dispMaps = [dispMaps]
    dispsFiltered = []
    i = 0
    for disp in dispMaps:
        if method == "wmf":
            # ignores invalid values
            i += 1
            dispFilt = wm.wmf(disp, window, sigma)
        elif method == "bilateral":
            # does not ignore invalid values
            i += 1
            invalid = np.min(disp)
            dispNormed = disp - invalid
            maskInvalid = dispNormed != 0
            dispBf = cv2.bilateralFilter(dispNormed, window, sigma, sigma)
            dispTemp = np.multiply(dispBf, maskInvalid)
            dispFilt = dispTemp + invalid
        elif method == "gauss":
            # does not ignore invalid values
            i += 1
            invalid = np.min(disp)
            dispNormed = disp - invalid
            maskInvalid = dispNormed != 0
            dispBf = cv2.GaussianBlur(dispNormed, (window, window), 0)
            dispTemp = np.multiply(dispBf, maskInvalid)
            dispFilt = dispTemp + invalid
        else:
            # does not ignore invalid values
            i += 1
            dispFilt = ndimage.median_filter(disp, size=window)
        dispsFiltered.append(dispFilt)
    return dispsFiltered


def depth_from_disparity(disp_maps, inverted, px_constant, tilt=5):
    """
    depth computation based on simplified trignometric equations and know
    tilting angles for eucentric tilting motions
     inputs: disparity_maps      - list of 'number of stereo-pair images' entries
                                   (f.i. 5 for 5 stereo-pair images)
                                   each entry contains 2 numpy arrays
                                   [n][0] unmasked disparity map (uxv)
                                   [n][1] 3xp array with masked x,y-coordinates
                                          and corresponding disparity-values
                                          in pixels (invalid pixels are not contained)
              px_constant        - px_constant to convert px values to
                                   metric values
              tilt               - tilt of between the stereo-pair images
     outputs: point_clouds       - list of 'number of stereo-pair images' entries
                                   (f.i. 5 for 5 stereo-pair images)
                                   each entry contains 1 numpy array of the
                                   form (3xi) where i is the number of points
                                   [x y z] for every point
    """
    p = px_constant
    point_clouds = []
    for n in range(len(disp_maps)):
        disp_map = disp_maps[n]
        # cull all bad pixels of the disp array to get the mask for the coordinates
        mask = disp_map != np.min(disp_map)
        disp_masked = disp_map[mask]
        # mask the x- and y-coords
        y_coords, x_coords = np.nonzero(mask)
        # tilt angle in radian
        tilt_rad = 2 * np.pi * tilt / 360
        # height from tilt using standard formula
        # scaling for visualization
        height_inv = np.squeeze((disp_masked * p) / (2 * np.sin(tilt_rad / 2)))
        # Spiegelung von height an xy-Ebene
        if inverted[n] == 0:
            z = -height_inv
        else:
            z = height_inv
        # zugehörige Koordinaten
        y = np.squeeze(y_coords * p)
        x = np.squeeze(x_coords * p)
        point_clouds.append(np.array((x, y, z)))
    point_clouds.append(p)
    return point_clouds


def height_plane_from_disparity_map(disp, k, p, tilt):
    """
    gets the plane of the heights of several chosen points in a disparity map

    inputs:    disp - nxm disparity map
               k    - number of used points/clicks
               p    - pixel constant
               tilt - tilt angle
    """
    # plot the disparity map
    plt.figure()
    plt.imshow(disp, "gray")
    plt.show()
    # plane points are switched in their x and y-coordinates because of ginput
    plane_points = plt.ginput(k)
    print("clicked", plane_points)
    # get tilt in radian
    tiltRad = 2 * np.pi * tilt / 360
    # create lists for equation system Ax=b
    tmp_A = []
    tmp_b = []
    # get height of clicked points
    for i in range(k - 1):
        # ginput switches x and y coordinates
        px = int(np.round(plane_points[i][1]))
        py = int(np.round(plane_points[i][0]))
        if disp[px, py] != -65:
            pxp = np.squeeze(px * p)
            pyp = np.squeeze(py * p)
            tmp_A.append([pxp, pyp, 1])
            # calculate plane for the computed height
            disp_plane = disp[px, py]
            # height from tilt using standard formula
            height_pts = -np.squeeze((disp_plane * p) / np.sin(tiltRad))
            tmp_b.append(height_pts)
    # determine the plane using the given points - simple least squares solution of Ax = b
    b = np.reshape((np.array(tmp_b)), (len(tmp_b), 1))
    A = np.array(tmp_A)
    # overdetermined system Ax=b solved by using pseudo inverse
    pinv = la.inv((np.transpose(A).dot(A))).dot(np.transpose(A))
    fit = pinv.dot(b)
    errors = b - A.dot(fit)
    residual = np.linalg.norm(errors) / len(errors)
    print("solution:")
    print("%f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
    print("errors:{}".format(errors))
    print("mean residual {}:".format(residual))
    return fit, residual
