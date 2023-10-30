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
#                   sequential processing of the reconstruction routine
#                   using AffineMultiViewStereo objects and a PointCloud object
#                   """

# packages
import numpy as np
import matplotlib.pyplot as plt
import sem3d.utils.preprocessing as pp
import sem3d.classes.AffineMultiViewStereo as amvs
from sem3d.classes.PointCloud import PointCloud
from sem3d.config.config import set_config
import matplotlib
matplotlib.use('TkAgg')


def main():

    # plt.ion()
    print("Starting the pipeline...")

    # configuration of the amvs-Pipeline -> config.py
    config = set_config()

    # load the images, get the stereoPairs of adjacent images and the stereo pair used for the triangulation
    stereoPairs, selectedStereoPairs = pp.get_images(config)

    # create two objects of type AffineMultiViewStereo
    MvsSequence = amvs.AffineMultiViewStereo.general_init(stereoPairs, config)
    MvsSelectedPairs = amvs.AffineMultiViewStereo(selectedStereoPairs, config)

    # process the sequence to estimate the camera parameters
    MvsSequence.detect_match_features()
    MvsSequence.apply_motion_constraints()
    MvsSequence.robust_epipolar_geometry()
    MvsSequence.optimize_epipolar_geometry()
    MvsSequence.measurement_matrix()
    MvsSequence.factorization()

    # process selected stereo pairs to compute the disparity maps
    MvsSelectedPairs.detect_match_features()
    MvsSelectedPairs.apply_motion_constraints()
    MvsSelectedPairs.robust_epipolar_geometry()
    MvsSelectedPairs.optimize_epipolar_geometry()
    MvsSelectedPairs.rectify_stereo_pairs()
    MvsSelectedPairs.disparity_maps()

    # segment rectified images
    if config['segmentation'] == True:
        MvsSelectedPairs.segment_rectified_images_unet()
        MvsSelectedPairs.segment_rectified_images_grabcut()

    # post process disparity map
    MvsSelectedPairs.post_process_disparity_maps()

    # copy dense matches attribute to the sequence object
    MvsSequence.denseMatches = MvsSelectedPairs.denseMatches

    # compute dense point clouds
    MvsSequence.compute_dense_point_clouds()

    # create PointCloud object to process the point clouds
    MvsPointClouds = PointCloud.from_mvs_object(MvsSequence)

    if config['densityFilter'] == True:
        MvsPointClouds.density_filter(
            numClusters=config['numClusters'], epsilon=12.5, minPoints=25)
    else:
        MvsPointClouds.visualize_point_cloud()

    if config['savePointCloud'] == True:
        print('Save point cloud to file..')
        if config['densityFilter'] == True:
            np.save('reconstructed_point_cloud.npy',
                    MvsPointClouds.filtDenseClouds[0])
        else:
            np.save('reconstructed_point_cloud.npy',
                    MvsPointClouds.densePointClouds[0])

    print('Reconstruction pipeline finished.')


if __name__ == '__main__':
    main()
