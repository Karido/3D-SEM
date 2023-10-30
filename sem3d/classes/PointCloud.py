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
#                   point cloud class to store the point clouds and perform
#                   initial processing steps
#                   """

# libraries
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# packages
import sem3d.visualization.output_visualization as vis


class PointCloud:

    def __init__(self, densePointClouds=None, sparsePointCloud=None, config=None):
        self.densePointClouds = densePointClouds
        self.sparsePointCloud = sparsePointCloud
        self.config = config

    @classmethod
    def from_mvs_object(cls, MvsClass):
        return cls(MvsClass.densePointClouds, MvsClass.sparsePointCloud, MvsClass.config)

    def mirror_dense_clouds(self, idx):

        Tmir = -np.eye(3)

        for i, X in enumerate(self.densePointClouds):
            if idx[i] == True:
                self.densePointClouds[i] = X@Tmir

        return self

    def mirror_sparse_cloud(self):

        Tmir = -np.eye(3)
        self.sparsePointCloud = self.sparsePointCloud@Tmir
        return self

    def density_filter(self, numClusters=5, epsilon=12.5, minPoints=25):
        self.filtDenseClouds = []
        for i, cloud in enumerate(self.densePointClouds):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud)
            labels = np.array(pcd.cluster_dbscan(
                eps=epsilon, min_points=minPoints, print_progress=True))
            max_label = labels.max()
            colors = plt.get_cmap("tab20")(
                labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

            # draw clusters
            vis.custom_draw_geometry(pcd)

            # keep numClusters largest clusters
            pcdArr = np.asarray(pcd.points)
            tmpPcdList = []
            for i in range(numClusters):
                tmpPcdList.append(pcdArr[labels == i])
            pcdFiltArr = np.vstack((tmpPcdList))

            pcdFilt = o3d.geometry.PointCloud()
            pcdFilt.points = o3d.utility.Vector3dVector(pcdFiltArr)
            self.filtDenseClouds.append(pcdFiltArr)
            vis.custom_draw_geometry(pcdFilt)

            if self.config['print'] == True:
                print("Point Cloud {} has {} clusters.".format(i+1, max_label + 1))

    def visualize_point_cloud(self):
        for i, cloud in enumerate(self.densePointClouds):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud)
            vis.custom_draw_geometry(pcd)
