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
#                   AffineMultiViewStereo Class which contains the attributes
#                   and sequential processing methods used in the
#                   reconstruction routine
#                   """


# libraries
import os
import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import rc
from PIL import Image

# packages
import sem3d.utils.preprocessing as pp
import sem3d.utils.utils as utils
import sem3d.feature_matching.feature_matching as fm
import sem3d.epipolar_geometry.affine_epipolar_geometry_algorithms as aeg
import sem3d.epipolar_geometry.distances as dist
import sem3d.epipolar_geometry.rectification as rec
import sem3d.factorization.factorization as fac
import sem3d.segmentation.unet_x32_softmax as unet
import sem3d.segmentation.segmentation as segmentation
import sem3d.segmentation.interactive_grabcut_segmentation as interactive_grabcut_segmentation
import sem3d.dense_matching.disparity_maps as dd
import sem3d.visualization.output_visualization as vis

#####################################


class AffineMultiViewStereo:

    def __init__(self, stereoPairs=None, config=None):
        self.cameraParameters = None
        self.centroids = None
        self.denseMatches = None
        self.densePointClouds = None
        self.densePointCloudsFilt = None
        self.disparityMaps = None
        self.epipolarSets = None
        self.grabCutMasks = None
        self.matches = None
        self.matchesAllViews = None
        self.measurementMatrix = None
        self.config = config
        self.postProcDisparityMaps = None
        self.predictedMasks = None
        self.rectificationData = None
        self.sparsePointCloud = None
        self.stereoPairs = stereoPairs

    @classmethod
    def general_init(cls, stereoPairs, config):
        return cls(stereoPairs, config)

    def detect_match_features(self):
        """
        wrapper function to perform the matching based on the given config

        stereoPairs - list of lists containing the two images of each stereo pair
        config     - dict with flags and parameters

        --------------------------------------------------
        matches     - list of dicts containing the matched features as pts1 and pts2 
        """

        # determine dataType to select appropriate matching methods
        if self.config['featureMethod'] in ('AKAZE', 'BRISK'):
            dataType = 'binary'
        elif self.config['featureMethod'] == 'ORB':
            dataType = 'binaryORB'
        else:
            dataType = 'standard'

        self.matches = []
        for i, pair in enumerate(self.stereoPairs):

            # keypoint and descriptor creation
            kp1, des1, kp2, des2 = fm.detect_features(
                pair, self.config['featureMethod'], self.config['keypointThresh'])

            # descriptor matching method
            if self.config['matchingMethod'] == 'flann':
                pts1, pts2 = fm.flann_matching(
                    des1, des2, kp1, kp2, dataType, self.config['matchingThresh'])
            else:
                pts1, pts2 = fm.brute_force_matching(
                    des1, des2, kp1, kp2, dataType, self.config['matchingThresh'])

            mp = {}
            mp['pts1'] = pts1
            mp['pts2'] = pts2
            mp['n'] = len(pts1)
            mp['n_kp'] = len(kp1)
            self.matches.append(mp)

        if self.config['print'] == True:
            print('\nNumber of Matches - {} - {}: '.format(self.config['featureMethod'],
                                                           self.config['matchingMethod']),
                  end='', flush=True)
            [print('{} '.format(m['n']), end='', flush=True)
             for m in self.matches]
            print('\n')

        return self

    def apply_motion_constraints(self):
        """
        wrapper function to be used in the reconstruction routine
        1) remove outliers based on u- and v-motion
        2) remove multiple matched matches

        matches     - list of dicts containing the matched features as pts1 and pts2 
        config     - dict with set flags
        --------------------------------------------------
        matches     - list of dicts containing the matched features as pts1 and pts2 
        """

        if self.config['motionConstraints'] == True:
            numOutU = []
            numOutV = []
            for j, m in enumerate(self.matches):
                m, outU1, outU2 = fm.v_motion_check(
                    m, th=self.config['vMotionTh'])
                numOutV.append(len(outU1))

            for j, m in enumerate(self.matches):
                m, outV1, outV2 = fm.u_motion_check(
                    m, th=self.config['uMotionTh'])
                numOutU.append(len(outV1))

            if self.config['print'] == True:
                print('Matches removed due to u-motion: ', end='', flush=True)
                [print('{} '.format(number), end='', flush=True)
                 for number in numOutU]
                print('\nMatches removed due to v-motion: ', end='', flush=True)
                [print('{} '.format(number), end='', flush=True)
                 for number in numOutV]

        else:
            pass

        # remove duplicates
        if self.config['removeMultiMatches'] == True:
            numOutM = []
            for j, m in enumerate(self.matches):
                m['pts1'], m['pts2'], outMpts, _ = fm.remove_multiple_matched_pts(
                    m['pts1'], m['pts2'])
                numOutM.append(len(outMpts))

            if self.config['print'] == True:
                print('\nMultiple matched features removed: ', end='', flush=True)
                [print('{} '.format(number), end='', flush=True)
                 for number in numOutM]
                print('\n')

        return self

    def robust_epipolar_geometry(self):  # matches, size, config):
        """
        main function to perform the robust estimation of the fundamental matrix
        using the affine gold standard algorithm and a robust estimation routine
        computed based on tensors for faster proccesing

        matches         - list dicts containing the 2D point coordinates
                          of matched features
        config         - dict
        --------------------------------------------------
        epiSets         - if nSets > 1: list of lists containing dictionaries
                          if nsets = 1: list of dictionaries dict 
                          dicts contain matches as pts1 and pts2, the 
                          computed Fa and the robustly estimated standard 
                          deviation of the residuals
        """

        # numSamples = self.config['numSamples']
        # nSets      = self.config['nSets']
        imageSize = np.shape(self.stereoPairs[0][0])[0]

        self.epipolarSets = []
        # len(matches) equals number of stereo pairs
        # compute robust Fa for every stereo pair
        for i, match in enumerate(self.matches):

            pts1 = match['pts1']
            pts2 = match['pts2']

            if len(pts1) < 4:
                raise Exception(
                    'Not enough matches to compute Fa. Reperform the matching for stereo pair {}.'.format(i+1))
            else:
                pass
            # fSets are the n best sets of the SAC routine
            fSets = aeg.estimate_robust_affine_fundamental_matrix(
                pts1, pts2, self.config['numSamples'], imageSize, self.config)

            # get the number of inliers
            inMax = fSets[0][0]

            # estimate the number of numSamples to achieve a probability of 99% for a good set
            eps = 1-len(inMax)/len(pts1)
            # pGood = 1-(1-(1-eps)**4)**num_samples, underlying formula
            pGood = 0.99
            ns = np.log(1-pGood)/(np.log(1-(1-eps)**4))

            # add a margin of 500 sets to account for degenerate sets (task is not time sensitive)
            # and check if more samples are needed
            if (ns+500) > self.config['numSamples']:
                if np.round(ns) > 10000:
                    raise Exception('Too much outliers in the detected matches for stereo pair {}.\n'
                                    'Number of computed sample capped to 10000.'.format(i+1))
                    ns = 10000
                else:
                    ns = np.round(ns)+500
                    fSets = aeg.estimate_robust_affine_fundamental_matrix(
                        pts1, pts2, ns, imageSize, self.config)
            else:
                pass

            # get the nSets best sets
            nSets = []
            for n in range(self.config['nSets']):
                npts1 = fSets[n][0]
                npts2 = fSets[n][1]
                bestF = fSets[n][2]
                cost = fSets[n][3]
                std = fSets[n][-1]
                epiSet = {}
                epiSet['cost'] = cost
                epiSet['F'] = bestF
                epiSet['pts1'] = npts1
                epiSet['pts2'] = npts2
                epiSet['std'] = std
                nSets.append(epiSet)

            # in normal mode, when only the best sample is needed, directly append dict
            if self.config['nSets'] == 1:
                self.epipolarSets.append(epiSet)
            else:
                self.epipolarSets.append(nSets)

            if self.config['print'] == True:
                # compute and print the residuals for the best F and the corresponding inliers as the rmse
                if self.config['costFunction'] == 'sampson':
                    _, _, rmseResiduals, _ = dist.sampson_distance(
                        nSets[0]['pts1'], nSets[0]['pts2'], nSets[0]['F'])
                elif self.config['costFunction'] == 'symEpi':
                    _, _, rmseResiduals, _ = dist.sym_epi_dist(
                        nSets[0]['pts1'], nSets[0]['pts2'], nSets[0]['F'])
                print('{} / {} ({:.2f}% inliers), '.format(len(inMax),
                                                           len(pts1),
                                                           (len(inMax)/len(pts1))*10**2),
                      end='', flush=True)
                print('RMSE({}) = {:.4f} px²\n'.format(
                    self.config['costFunction'], rmseResiduals))

        return self

    def optimize_epipolar_geometry(self):
        """
        perform an iterative local optimization over all inlying matches

        matches        - list of dicts containing the matches for all stereo pairs
        epiSetsAffine  - list of dicts containing F and corresponding inliers
                         computed via the chosen SAC scheme for each stereo pair
        config        - config['estimationSigma']
                         config['costFunction']
                         config['fixedSigma']
                         config['kappa']
                         config['searchRange']
                         config['robustMethod']
        --------------------------------------------------
        epiSetsAffine  - epiSets with updated F and inliers after the local optimization

        """

        for j, epiSet in enumerate(self.epipolarSets):

            npts1i, npts2i, Fi, i, rhoi, sigi = aeg.local_opt_affine(
                self.matches[j], epiSet, self.config)

            # select best result yielding the lowest cost/best score, perform local opt at least once
            if self.config['robustMethod'] == 'RANSAC':
                idxBestScore = np.argmax(rhoi[1:])+1
            else:
                idxBestScore = np.argmin(rhoi[1:])+1

            # update the fundamental matrix and corresponding inliers
            epiSet['F'] = Fi[idxBestScore]
            epiSet['pts1'] = npts1i[idxBestScore]
            epiSet['pts2'] = npts2i[idxBestScore]

            if self.config['print'] == True:
                # compute and print the rmse errorArr of the chosen cost computed for all inliers
                if self.config['costFunction'] == 'sampson':
                    _, _, rmseResiduals, _ = dist.sampson_distance(
                        epiSet['pts1'], epiSet['pts2'], epiSet['F'])
                elif self.config['costFunction'] == 'symEpi':
                    _, _, rmseResiduals, _ = dist.sym_epi_dist(
                        epiSet['pts1'], epiSet['pts2'], epiSet['F'])
                print('iters = {}, score/cost - {}: {:.4f} -> GM: {:.4f};\n'.format(idxBestScore,
                      self.config['robustMethod'], rhoi[0], rhoi[idxBestScore]), end='', flush=True)
                print('{:<24} RMSE({}) = {:.4f} px², {} matches'.format(
                    ' ', self.config['costFunction'], rmseResiduals, len(epiSet['pts1'])))

        return self

    def measurement_matrix(self):
        """
        iterates over all matches
        if in previous and next image matches are found, the feature point 
        is saved and compared to the matches in the next image

        input:  epiSetsAffine - sets containing fundamental matrices and point correspondences
                matches - matches between stereo images([image1,image2],[image2,image3],...)
                          list of n dict, containing 2 numpy arrays each
                          with image coordinates of (descriptor) matched
                          feature points, keys: pts_1 and pts_2

        output: measurement matrix with registered point correspondences

        """

        matchesTmp = []
        for i in range(0, len(self.epipolarSets)):
            matchesTmp.append(self.epipolarSets[i])

        self.measurementMatrix, self.centroids, self.matchesAllViews, _ = fac.build_measurement_matrix(
            matchesTmp)

        if self.config['print'] == True:
            print("\nMeasurement Matrix contains {} matches over {} views.\n".format(np.max(
                np.shape(self.measurementMatrix)), int(np.min(np.shape(self.measurementMatrix))/2)))

        return self

    def factorization(self):

        def perform_factorization(W, config):
            # helper function to use the respective factorization method
            if config['factorizationMethod'] == 'SC':
                result = fac.factorization_sc(
                    W, normRot=config['normalizedView'])
            elif config['factorizationMethod'] == 'OR':
                result = fac.factorization_or(W)
            return result

        W = self.measurementMatrix

        resultFactor = perform_factorization(
            self.measurementMatrix, self.config)

        reproW, _ = fac.reprojected_measurement_matrix(resultFactor)
        reproErrW = fac.signed_error(reproW, self.measurementMatrix)

        if self.config['reproErrorTh'] > 0:
            self.measurementMatrix, _ = fac.remove_outliers_repro(
                W, reproW, self.config['reproErrorTh'])

            # correct the registration and translate the centroid in the origin
            self.measurementMatrix = self.measurementMatrix - \
                np.vstack(np.mean(self.measurementMatrix, axis=1))

            # correct the previously computed centroid
            self.centroids = self.centroids + np.hstack((np.vstack(np.mean(self.measurementMatrix, axis=1))[:int(len(
                self.measurementMatrix)/2)], np.vstack(np.mean(self.measurementMatrix, axis=1))[int(len(self.measurementMatrix)/2):]))

            # update result
            resultFactor = perform_factorization(
                self.measurementMatrix, self.config)

            reproNewW, _ = fac.reprojected_measurement_matrix(resultFactor)
            reproErrNewW = fac.signed_error(reproNewW, self.measurementMatrix)

        camDict = {}
        camDict['rotations'] = np.asarray(resultFactor[0])
        camDict['intrinsics'] = [resultFactor[2]/resultFactor[2]
                                 [self.config['normalizedView']], resultFactor[3]]
        self.cameraParameters = camDict
        self.sparsePointCloud = np.asarray(resultFactor[1])

        if self.config['visualize'] == True:
            # plot reprojection errors
            vis.plot_reprojection_error(reproErrW,
                                        x=[-1.5, 1.51, 1.5],
                                        y=[-1.5, 1.51, 1.5],
                                        title='Reprojektionsfehler Initial')
            if self.config['reproErrorTh'] > 0:
                vis.plot_reprojection_error(reproErrNewW,
                                            x=[-1.5, 1.51, 1.5], y=[-1.5, 1.51, 1.5],
                                            title='Reprojektionsfehler Thresholded')

        if self.config['print'] == True:

            print('\n{} matche(s) removed due to the reprojection error threshold'
                  ' of {} px.'.format((np.shape(W)[1]-np.shape(self.measurementMatrix)[1]), self.config['reproErrorTh']))

            # print angles, scaling factors and RMSE
            vis.print_rotation_angles(
                resultFactor[0], name=self.config['factorizationMethod'])
            print('Relative scaling factors of each view:'
                  '\n{}\n'.format(
                      resultFactor[2]/resultFactor[2][self.config['normalizedView']]),
                  end='', flush=True)
            print('RMSE of the reprojection error is given as '
                  '{:.4f} px.'.format(utils.compute_rmse(reproErrW.flatten())))

        return self

    def rectify_stereo_pairs(self):
        """
        wrapper function which performs the rectification based on the 
        user specified config

        spDense                 - list containing a list with the two images 
                                  of the stereo pair as uint8 numpy arrays
        epiSetsAffineSpDense    - list containing a dict with 'F', pts1, pts2
        config                 - dict
        """

        self.rectificationData = []

        for i in range(len(self.stereoPairs)):

            if self.config['rectificationMethod'] == "rigid":

                # compute rectifying transformations
                T1, T2 = rec.rectifying_rigid_transformations(self.epipolarSets[i]['F'],
                                                              utils.homogenize(
                                                                  self.epipolarSets[i]['pts1']).T,
                                                              utils.homogenize(self.epipolarSets[i]['pts2']).T)

            elif self.config['rectificationMethod'] == "similarity":

                # compute rectifying transformations
                T1, T2 = rec.rectifying_similarities(self.epipolarSets[i]['F'])

            elif self.config['rectificationMethod'] == "affine":

                # compute rectifying transformations
                T1, T2 = rec.affine_rectification(self.epipolarSets[i]['F'],
                                                  utils.homogenize(
                                                      self.epipolarSets[i]['pts1']),
                                                  utils.homogenize(self.epipolarSets[i]['pts2']))

            elif self.config['rectificationMethod'] == "quasiEuclidean":

                # compute rectifying transformations
                qerData = rec.quasi_euclidean_rectification(self.stereoPairs,
                                                            self.epipolarSets,
                                                            focal=self.config['rectificationFocal'][0],
                                                            fixedFocal=self.config['rectificationFocal'][1])
                T1, T2 = qerData[0]['H1'], qerData[0]['H2']

            else:
                print('No method was choosen from rigid, similarity, affine and quasiEuclidean. '
                                'Similarity is used as default.')
                # compute rectifying transformations
                T1, T2 = rec.rectifying_similarities(self.epipolarSets[i]['F'])

            # apply rectifying transformations to stereoPair
            img1r, img2r, Ht = rec.transform_two_images_bb(
                T1, T2, self.stereoPairs[i][0], self.stereoPairs[i][1])

            # determine full transformation
            H1 = Ht.dot(T1)
            H2 = Ht.dot(T2)

            # transform matches into the rectified image
            pts1r = utils.homogenize(rec.transform_2D(
                utils.homogenize(self.epipolarSets[i]['pts1']).T, H1)).T
            pts2r = utils.homogenize(rec.transform_2D(
                utils.homogenize(self.epipolarSets[i]['pts2']).T, H2)).T

            rectDict = {}
            rectDict['img1r'] = np.uint8(img1r)
            rectDict['img2r'] = np.uint8(img2r)
            rectDict['Ht'] = Ht
            rectDict['T1'] = T1
            rectDict['T2'] = T2
            self.rectificationData.append(rectDict)

            if self.config['print'] == True:
                print('\nRectification of stereo pair {} via {}.. \n'.format(
                    i+1, self.config['rectificationMethod']), end='', flush=True)
                # compute the pre and post rectification v displacement between matches
                vDisplacementPost = pts1r[:, 1] - pts2r[:, 1]
                vDisplacementPre = self.epipolarSets[i]['pts1'][:,
                                                                1] - self.epipolarSets[i]['pts2'][:, 1]
                preRectRmse = utils.compute_rmse(vDisplacementPre)
                postRectRmse = utils.compute_rmse(vDisplacementPost)
                print('\tRMSE of the v disparities between the matches:\n'
                      '\tPreRectification:  {:.4f} px\n'
                      '\tPostRectification: {:.4f} px\n'.format(preRectRmse, postRectRmse))

            if self.config['visualize'] == True:
                # visualize original and rectified stereo pairs plus epipolar lines
                Fr = np.matrix([[0, 0, 0], [0, 0, -1], [0, 1, 0]])

                vis.epilines_matches(self.epipolarSets[i]['pts1'],
                                     self.epipolarSets[i]['pts2'],
                                     self.epipolarSets[i]['F'],
                                     self.stereoPairs[i][0],
                                     self.stereoPairs[i][1],
                                     "Non-Rectified Stereo Pair",
                                     divide=10)

                vis.epilines_matches(pts1r,
                                     pts2r,
                                     Fr,
                                     img1r,
                                     img2r,
                                     "Rectified Stereo Pair",
                                     divide=10)

        return self

    def disparity_maps(self):

        self.disparityMaps = []

        for i in range(len(self.rectificationData)):

            if self.config['rankTransform'] == True:
                img1, img2 = pp.rank_transform_rectified_images(
                    self.rectificationData[i], window=15)
            else:
                img1, img2 = self.rectificationData[i]['img1r'], self.rectificationData[i]['img2r']

            # pad images with zeros so sgbm can work properly without creating black edges in the ROI
            img1p = np.pad(
                img1, self.config['paddedZeros'], 'constant', constant_values=0)
            img2p = np.pad(
                img2, self.config['paddedZeros'], 'constant', constant_values=0)

            # perform a dense semi global matching procedure with speckle filtering of the disparity map
            dispSGBM = dd.sgbm(img1p, img2p, *self.config['sgbmParams'])

            # true disparity
            self.disparityMaps.append(dispSGBM.astype(np.float32) / 16.0)

        return self

    def segment_rectified_images_unet(self):

        self.predictedMasks = []

        for rectSet in self.rectificationData:

            # input shape of the UNet
            img1 = Image.fromarray(rectSet['img1r'])
            img1Resized = img1.resize(
                self.config['unetInputShape'], resample=Image.BICUBIC)

            # generate input for the NN
            imgSet = np.zeros((1,
                               self.config['unetInputShape'][0],
                               self.config['unetInputShape'][1], 1),
                              dtype=np.float32)
            imgSet[0, :, :, 0] = img1Resized

            model = unet.unet_model()
            weightsData = 'x32_128_no_weightmap_epoch_37_weights.h5'

            model.load_weights(os.path.join('res/weights', weightsData))

            # get the prediction (mask)
            predict = model.predict([imgSet], verbose=1, batch_size=1)

            # get a binary mask
            maskBin = segmentation.get_binary_mask(
                predict[0, :, :, 1], thresh=self.config['binarizationThUnet'])
            imgMask = Image.fromarray(maskBin)
            self.predictedMasks.append(imgMask)

            if self.config['visualize'] == True:
                # plot the images
                plt.figure(figsize=(8, 6))
                plt.axis('off')
                # plt.rcParams.update({'font.size': 8})
                rc('font', size=8)
                rc('font', family='Segoe UI')
                plt.subplot(221), plt.imshow(
                    img1, 'gray'), plt.title("original image")
                plt.subplot(222), plt.imshow(
                    img1Resized, 'gray'), plt.title("resized image")
                plt.subplot(223), plt.imshow(
                    predict[0, :, :, 1], 'gray'), plt.title("predicted mask")
                plt.subplot(224), plt.imshow(
                    maskBin, 'gray'), plt.title("binarized mask")
                plt.show()

            if self.config['print'] == True:
                print('Pre-Segmentation via UNet.\n', end='', flush=True)

        return self

    def segment_rectified_images_grabcut(self):

        # iterate over all entries of stereo pairs
        self.grabCutMasks = []
        for i, rectSet in enumerate(self.rectificationData):

            # input shape of the UNet
            img1 = rectSet['img1r']

            # check is there is a predicted mask
            if (self.predictedMasks is None):
                pass
            else:
                # first GrabCut is performed using images of the size of the UNet input
                grabCutMaskInit, inputGrabCutInit = segmentation.grab_cut_refinement(self.predictedMasks[i],
                                                                                     Image.fromarray(
                                                                                         img1),
                                                                                     True,
                                                                                     self.config['maskWidths'][0])

                # for the second grabCut iteration, resize original image, when the flag is true
                if self.config['resizeImage'] == True:
                    newInput = cv2.resize(cv2.bitwise_not(grabCutMaskInit),
                                          self.config['sizeSegmentedImage'],
                                          interpolation=cv2.INTER_CUBIC)
                else:
                    newInput = cv2.bitwise_not(grabCutMaskInit)

                # second time, the original or resized image is used to perform GrabCut with thinner regions of probable fore- and background
                grabCutMaskAuto, inputGrabCut2 = segmentation.grab_cut_refinement(Image.fromarray(newInput),
                                                                                  Image.fromarray(
                                                                                      img1),
                                                                                  self.config['resizeImage'],
                                                                                  self.config['maskWidths'][1])

            # check if only manual segmentation, automatic and manual segmentation or no segmentation is desired
            if (self.config['segmentationManual'] == True) and (self.predictedMasks is not None):
                # replace fixed back- and foreground by probable back- and foreground
                # to enable the interactive segmentation and changes in the result
                manualSegInitMask = copy.copy(inputGrabCut2)

                manualSegInitMask[manualSegInitMask == 1] = 3
                manualSegInitMask[manualSegInitMask == 0] = 2

                manualSegInputImage = np.uint8(cv2.resize(img1,
                                                          np.shape(
                                                              manualSegInitMask),
                                                          interpolation=cv2.INTER_CUBIC))

                grabCutMask = interactive_grabcut_segmentation.grab_specimen(manualSegInputImage,
                                                                             manualSegInitMask,
                                                                             initByMaskInput=True)
                self.grabCutMasks.append(grabCutMask)

            elif (self.config['segmentationManual'] == False) and (self.predictedMasks is not None):
                self.grabCutMasks.append(grabCutMaskAuto)

            else:
                manualSegInputImage = np.uint8(cv2.resize(img1,
                                                          self.config['sizeSegmentedImage'],
                                                          interpolation=cv2.INTER_CUBIC))

                grabCutMask = interactive_grabcut_segmentation.grab_specimen(manualSegInputImage,
                                                                             None,
                                                                             initByMaskInput=False)
                self.grabCutMasks.append(grabCutMask)

        # plot the masks dependent on the chosen option
        if self.config['visualize'] == True:
            fig = plt.figure()
            rc('font', size=10)
            rc('font', family='Segoe UI')
            plt.rcParams.update({'font.size': 8})
            plt.title(
                "Plots for the first image of the last selected stereo pair")
            plt.axis('off')
            try:
                inverted_mask = (np.full((128, 128), 255) -
                                 self.predictedMasks[i])
                plt.subplot(245), plt.imshow(inverted_mask, 'gray'), plt.title(
                    "output mask UNet"), plt.axis('off')
                plt.subplot(241), plt.imshow(img1, 'gray'), plt.title(
                    "original image"), plt.axis('off')
                plt.subplot(242), plt.imshow(inputGrabCutInit, 'gray'), plt.title(
                    "Input mask GC Init"), plt.axis('off')
                plt.subplot(246), plt.imshow(grabCutMaskInit, 'gray'), plt.title(
                    "Output mask GC Init"), plt.axis('off')
                plt.subplot(243), plt.imshow(inputGrabCut2, 'gray'), plt.title(
                    "Input mask GC Final"), plt.axis('off')
                plt.subplot(247), plt.imshow(grabCutMaskAuto, 'gray'), plt.title(
                    "Output mask GC Final"), plt.axis('off')
                try:
                    if self.config['segmentationManual'] == True:
                        plt.subplot(244), plt.imshow(manualSegInitMask, 'gray'), plt.title(
                            "Input to Manual Segmentation"), plt.axis('off')
                        plt.subplot(248), plt.imshow(grabCutMask, 'gray'), plt.title(
                            "Manually Segmented Mask"), plt.axis('off')
                except:
                    pass
                fig.set_size_inches(10, 6)
                rc('font', size=10)
                rc('font', family='Segoe UI')
                plt.show()
            except:
                plt.subplot(211), plt.imshow(img1, 'gray'), plt.title(
                    "Original Image"), plt.axis('off')
                plt.subplot(212), plt.imshow(grabCutMask, 'gray'), plt.title(
                    "Manually Segmented Mask"), plt.axis('off')
                fig.set_size_inches(8, 6)
                rc('font', size=10)
                rc('font', family='Segoe UI')
                plt.show()

        if self.config['print'] == True:
            print('Post-Segmentation via GrabCut.\n', end='', flush=True)

        return self

    def post_process_disparity_maps(self):
        # TO ADD: interpolation of the disparity maps in the masked regions
        #        (weighted) median, biliteral filter operations
        if self.config['print'] == True:
            print('Post Processing of Disparity Maps..\n', end='', flush=True)
            print('Mask Disparity Maps and extract dense matches in original stereo pair.\n',
                  end='', flush=True)

        self.postProcessedDisparityMaps = []
        self.denseMatches = []

        for i, dispMap in enumerate(self.disparityMaps):
            if self.config['segmentation'] == True:
                dispMapMasked = segmentation.mask_disp(
                    dispMap, self.grabCutMasks[i], padded_zeroes=self.config['paddedZeros'], plot=self.config['visualize'])
            else:
                ones_mask = np.ones((np.shape(dispMap)[
                                    0] - 2*self.config['paddedZeros'], np.shape(dispMap)[1] - 2*self.config['paddedZeros']))
                dispMapMasked = segmentation.mask_disp(
                    dispMap, ones_mask, padded_zeroes=self.config['paddedZeros'], plot=self.config['visualize'])

            self.postProcessedDisparityMaps.append(dispMapMasked)

            self.denseMatches.append(utils.dense_matches_from_disparity(self.stereoPairs[0][0],
                                                                        dispMapMasked,
                                                                        self.rectificationData[i],
                                                                        img_base_right=False))
        return self

    def compute_dense_point_clouds(self):

        # iterate over all entries of self.config['selectedPairs']
        self.densePointClouds = []

        for i, idx in enumerate(self.config['selectedPairs']):

            # intrinsics
            k1 = self.cameraParameters['intrinsics'][0][idx[0]]
            k2 = self.cameraParameters['intrinsics'][0][idx[1]]
            c = self.cameraParameters['intrinsics'][1][0]
            s = self.cameraParameters['intrinsics'][1][1]

            # get truncated rotation matrices
            rot1 = self.cameraParameters['rotations'][idx[0]][:2]
            rot2 = self.cameraParameters['rotations'][idx[1]][:2]

            # build affine camera matrices
            P1 = k1*np.array([[c, 0], [s, 1]]).dot(rot1)
            P2 = k2*np.array([[c, 0], [s, 1]]).dot(rot2)

            # center matches / no need to further consider the translations
            matches1 = self.denseMatches[i][0][:, 0:2].T - \
                np.reshape(self.centroids[idx[0]], (2, 1))
            matches2 = self.denseMatches[i][1][:, 0:2].T - \
                np.reshape(self.centroids[idx[1]], (2, 1))

            # perform linear triangulation (inhomogeneous equation system)
            matches12 = np.vstack((matches1, matches2))
            P12 = np.vstack((P1, P2))
            X = utils.inhom_triangulation(matches12, P12)
            self.densePointClouds.append(X)

            # if self.config['visualize'] == True:
            #     vis.plot_pc(X.T, color = 'grey', ms = 0.1, sparse = 10)
            if self.config['print'] == True:
                print('Point Cloud {} computed.\n'.format(
                    i+1), end='', flush=True)

        return self
