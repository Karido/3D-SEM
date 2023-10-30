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
#                   define config for the sequential processing of the
#                   reconstruction routine, details below
#                   """

import os

def set_config():
    # define config
    config = {}
    
    ###########
    # general config
    ###########

    # path of the image folder
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    parent_directory = os.path.dirname(current_directory)
    parent_directory = os.path.dirname(parent_directory)
    used_directory = os.path.join(parent_directory, 'res/quarz') # or use dsa
    config["imageFolder"] = used_directory

    # visualization of pipeline interim results
    config["visualize"] = True

    # printing of status and results
    config["print"] = True

    # histogram matching of input images
    config["histMatch"] = True

    # segmentation of the specimen (more precisely "particle-like" specimen)
    # (if false the whole point cloud except the used padding is reconstructed)
    config["segmentation"] = True

    # apply density filter to point cloud
    config["densityFilter"] = True

    # save point cloud in the specified directory
    # (point cloud is otherwise only stored in the MvsPointClouds object)
    config["savePointCloud"] = False

    ###########
    # feature detection and matching
    ###########

    # feature algorithm
    config["featureMethod"] = "ROOTSIFT"
    # most sensitive parameter of the feature algorithm to control the number of detected keypoints
    config["keypointThresh"] = 0.04
    # matching method, brute force (bf) or fast library of approximated nearest neighbors (flann)
    config["matchingMethod"] = "bf"
    # ratio or threshold according to D. Lowe to discard low quality matches, lower is more restrictive
    config["matchingThresh"] = 0.75
    # motion constraints can be used when orbital camera motions/ eucentric tiltings were used to acquire the images
    config["motionConstraints"] = True
    # thresholds of maximum allowed disparities between matched features in u- and v-direction
    config["uMotionTh"] = 500
    config["vMotionTh"] = 50
    # remove matches that have been matched multiple times (results in a set of unique matches)
    config["removeMultiMatches"] = True

    ###########
    # epipolar geometry
    ###########

    # robust error term: ransac, msac, lmeds, mlesac
    config["robustMethod"] = "MLESAC"
    # cost function: 'sampson' or 'symEpi'
    config["costFunction"] = "sampson"
    # pick points from which the fundamental matrix is estimated from different regions
    config["bucketing"] = False
    # estimate the standard estimation of the residuals robustly to determine a threshold
    config["stdEstimation"] = "adaptive"
    # when 'fixed', use a chosen value for the standard deviation
    config["stdFixed"] = 1
    # factor to compute the threshold as (kappa*std)**2
    config["kappa"] = 2.5
    # searchRange of corresponding features, parameter used in MLESAC to determine the cost for outliers
    config["searchRange"] = 200
    # number of samples intitially computed by the sample consensus (number of estimated fundamental matrices)
    config["numSamples"] = 1000
    # best nSets returned from the sample consensus based on the chosen robust method/error term
    config["nSets"] = 1

    ###########
    # factorization
    ###########

    # orthographic (OR) or scaled orthographic (SC) currently supported
    config["factorizationMethod"] = "SC"
    # world coordinate system is located at the camera coordinate system of the "normalized" view
    config["normalizedView"] = 0
    # matches that are removed, when the reprojection error is higher than the chosen threshold
    config["reproErrorTh"] = 1.0

    ###########
    # rectification and dense matching
    ###########

    # selected image pairs that are used to compute dense reconstructions (list of tuples of indexes)
    config["selectedPairs"] = [(0, 2)]  # ,(0,1)]
    # rectification method
    # "rigid", "similarity", "affine", "quasiEuclidean"
    config["rectificationMethod"] = "similarity"
    # When Quasi-Euclidean-Rectification method is used:
    # the focal length factor in computed in the range [-1.5, 1.5] via non-linear minimization 
    # of the Sampson distance, the focal length is further obtained by
    # factor*(imgWidth + imgHeight)
    # when rectificationFocal == True, the focal length factor is not optimized and fixed to
    # the specified value for instance 5*(imgWidth + imgHeight)
    config["rectificationFocal"] = (5, False)
    # apply rank transform to the images before performing the dense matching via SGBM
    config["rankTransform"] = False
    # pad image with zeroes to perform the dense matching successfully at the image borders
    config["paddedZeros"] = 200
    # parameters of semi global block matching
    config["sgbmParams"] = (-128, 128, 8, 32, 0, 15, 1, 175, 3, 1)
    # minDisp, maxDisp, P1factor, P2factor, preFilterCap, uniquenessRatio, dispDiff, speckleWindowSize, speckleRange,  windowSize
    # minDisp           -	Minimum possible disparity value. Normally, it is zero but sometimes rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.
    # maxDisp           -	Maximum  possible disparity value minus minimum disparity. The total number of disparity values must be divisible by 16.
    # P1	            -   The first parameter controlling the disparity smoothness. See below.
    # P2	            -   The second parameter controlling the disparity smoothness. The larger the values are, the smoother the disparity is. P1 is the penalty on the disparity change by plus or minus 1 between neighbor pixels. P2 is the penalty on the disparity change by more than 1 between neighbor pixels. The algorithm requires P2 > P1 . See stereo_match.cpp sample where some reasonably good P1 and P2 values are shown (like 8*number_of_image_channels*blockSize*blockSize and 32*number_of_image_channels*blockSize*blockSize , respectively) [P1factor = 8, P2factor = 32].
    # preFilterCap	    -   Truncation value for the prefiltered image pixels. The algorithm first computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval. The result values are passed to the Birchfield-Tomasi pixel cost function.
    # uniquenessRatio	-   Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough.
    # dispDiff   	    -   Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.
    # speckleWindowSize	-   Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
    # speckleRange	    -   Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
    # windowSize        -	Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.

    ###########
    # segmentation
    ###########

    config["unetInputShape"] = (128, 128)
    # threshold to binarize the softMax output (1 = background, 0 = foreground)
    # generally background is predicted slightly more secure
    config["binarizationThUnet"] = 0.66
    # assign two 'maskWidths' values, which are interpreted as
    # the absolute pixel width in the image when given as an integer
    # and otherwise, the percentage width of the input mask image is used,
    # the values define the width of the presegmented probable fore- and background
    config["maskWidths"] = (0.20, 0.02)
    # interactive segmentation based on the results of the initially performed automated segmentation,
    # otherwise the mask segemented by the U-Net is used to automatically create various probability "zones"
    # for fore- and background
    config["segmentationManual"] = False
    # temporarily resize the image to perform GrabCut,
    # if false original image size is used
    config["resizeImage"] = False
    config["sizeSegmentedImage"] = (1000, 1000)
    
    # point cloud density filtering and clustering, number of clusters kept
    config['numClusters'] = 3

    return config
