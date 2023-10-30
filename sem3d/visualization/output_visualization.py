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
#                   output and visualization in the reconstruction routine
#                   """


# libraries
import numpy as np
import sys
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.patches import ConnectionPatch
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection


# packages
import sem3d.utils.utils as utils


def show_img(img, cmap=0, vmi=None, vma=None):
    """
    function to plot a grayscale image in a matplot figure
    """
    plt.figure()
    plt.imshow(img, cmap='gray', vmin=vmi, vmax=vma)
    if cmap == 1:
        plt.colorbar()
    plt.show


def plot_matches(imgL, imgR, pts1, pts2, markersize=8, lines=1, n=1, c1='#22dc18', c2='#DC189B'):
    """
    plots the matches and draws connections between the matched features if
    lines are set as true
    inputs: imgL  - image
            imgR  - image
            pts1  - (nx2) keypoint array
            pts2  - (nx2) keypoint array
            lines - if true draw connections
    """

    fig = plt.figure(figsize=(12, 8))

    # create first subplot
    ax1 = fig.add_subplot(121)
    ax1.imshow(imgL, cmap='gray')

    # plot points in first image
    ax1.plot(pts1[::n, 0], pts1[::n, 1], '+', color=c1, ms=markersize)
    ax1.axis('off')

    # create second subplot
    ax2 = fig.add_subplot(122)
    ax2.imshow(imgR, cmap='gray')

    # plot points in second image
    ax2.plot(pts2[::n, 0], pts2[::n, 1], '+',
             fillstyle='none', color=c2, ms=markersize)
    ax2.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0)

    # create connections between points in the two subplots
    if lines == 1:
        for i in range(int(len(pts1)/n)):
            i = i*n
            xy1 = (pts1[i, 0], pts1[i, 1])
            xy2 = (pts2[i, 0], pts2[i, 1])
            # green '#71E979' # blue  '#27f1d6'
            con = ConnectionPatch(xyA=xy2, xyB=xy1, coordsA="data", coordsB="data",
                                  axesA=ax2, axesB=ax1, color='#27f1d6', linewidth=0.5, alpha=0.5)
            ax2.add_artist(con)

    plt.show()

    return None


def epilines_matches(pts1, pts2, F, img1, img2, title, divide=10, vmi=0, vma=255):
    """
    in this description l is meant for image 1 and r for image 2!
    plots the epipolar lines given point correspondences, the related 
    fundamental matrix and the images, the number of plotted lines is divided 
    by divide

    return
    yI, yII     - points of the epilines
    """

    if len(pts1[0]) < 3:
        ON = np.ones((len(pts1), 1))
        pts1 = np.concatenate((pts1, ON), 1)
        pts2 = np.concatenate((pts2, ON), 1)

    heigth1 = len(img1)
    length1 = len(img1[0])
    heigth2 = len(img2)
    length2 = len(img2[0])

    rc('font', size=12)
    rc('font', family='Segoe UI')

    # set up figure
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle(title)
    first = fig.add_subplot(121)
    plt.axis('off')
    # define colors and plot settings
    c1 = '#d64f4f'
    c2 = '#6ee0cb'
    markersize = 4

    # first subplot
    first.imshow(img1, cmap='gray',  vmin=vmi, vmax=vma)
    first.plot(pts1[:, 0], pts1[:, 1], '.', color=c1, ms=markersize)

    # second subplot
    second = fig.add_subplot(122)
    second.imshow(img2, cmap='gray', vmin=vmi, vmax=vma)
    second.plot(pts2[:, 0], pts2[:, 1], '.',
                fillstyle='none', color=c2, ms=markersize)

    # compute epilines second img
    epilines_1 = F.T @ pts2.T

    # compute epilines first img
    epilines_2 = F   @ pts1.T

    # plot epilines over the following range and points
    x1 = np.linspace(1, length1-2, 2000)
    x2 = np.linspace(1, length2-2, 2000)

    yI = []
    yII = []
    for i in range(0, len(pts1), divide):
        y1 = (-epilines_1[0, i] * x1 - epilines_1[2, i]) / epilines_1[1, i]
        mask_l1 = y1 <= heigth1-2
        mask_l2 = y1 >= 0
        mask_l = mask_l1 * mask_l2
        y2 = (-epilines_2[0, i] * x2 - epilines_2[2, i]) / epilines_2[1, i]
        mask_r1 = y2 <= heigth2-2
        mask_r2 = y2 >= 0
        mask_r = mask_r1 * mask_r2
        # , color = colors[2]) #color='#ff00ee'
        first.plot(x1[mask_l], y1[mask_l], '-', linewidth=1.5)
        # , color = colors[2]) #color='#ff00ee',
        second.plot(x2[mask_r], y2[mask_r], '-', linewidth=1.5)
        yI.append(y1)
        yII.append(y2)

    # show plot
    plt.axis('off')
    plt.show()

    return yI, yII


def print_rotation_angles(R, name='R', digits=4):

    print(
        '{} Factorization - Rotation Angles [phiX, phiY, phiZ]:'.format(name))

    i = 0
    rotAngles = []
    for r in R:
        phiTup = utils.rotation_angles_arctan(r)
        phiList = []

        for phii in phiTup:
            phiList.append(int(phii*10**digits) / 10**digits)

        print('Image {}: {}'.format(i, phiList))
        i = i+1
        rotAngles.append(phiList)

    return rotAngles


def plot_reprojection_error(errorArr,
                            marksersize=6,
                            x=[0, 1, 0.1],
                            y=[0, 1, 0.1],
                            alp=0.9,
                            facecolor='#ffffff',
                            title='Mittlerer Reprojektionsfehler'):

    # define color map
    viridisBig = cm.get_cmap('gray', 512)
    colorMap = ListedColormap(viridisBig(np.linspace(0.15, 0.9, 256)))

    # compute oriented distance
    oriDist = np.sqrt(errorArr[0]**2 + errorArr[1]**2)

    # set up figure
    fig1 = plt.figure()
    fig1.set_size_inches(6, 4.8)

    # draw circle with radius 1 in lightblue
    circles(0, 0, 1, c='lightblue', alpha=0.4, ec='none')

    # plot error values as scatterplot
    sc = plt.scatter(errorArr[0], errorArr[1], c=oriDist,
                     alpha=alp, s=marksersize, cmap=colorMap)

    # create colorbar and set ticks
    cb = plt.colorbar(sc)
    cb.set_ticks((0, np.max(oriDist)))

    # axis style assignments
    ax = plt.gca()
    ax.set_xlabel('u in px')
    ax.set_ylabel('v in px')
    ax.set_title(title)
    ax.set_facecolor(facecolor)

    # set up font
    rc('font', size=12)
    rc('font', family='Segoe UI')
    rc('axes', labelsize=12)

    # define ticks and limits and show the plot
    plt.xticks(np.arange(x[0], x[1], step=x[2]))
    plt.yticks(np.arange(y[0], y[1], step=y[2]))
    plt.xlim(x[0]+0.01, x[1]+0.01)
    plt.ylim(y[0]+0.01, y[1]+0.01)
    plt.show()

    return None


def plot_disparity_map(disp, dmin=20, dmax=-20, font='standard', color='rainbow', vmi=0, vma=255):
    fig = plt.figure()
    disp[disp == np.min(disp)] = np.nan
    cax = plt.imshow(disp, cmap=color, vmin=vmi, vmax=vma)
    cbar = fig.colorbar(cax, ticks=[dmin, dmax], boundaries=np.linspace(
        dmin, dmax, 200), shrink=0.7)
    cbar.ax.set_yticklabels([dmin, dmax])
    plt.show


def plot_pc(pA,
            color='r',
            sparse=500,
            ms=8,
            ds=False,
            azi=75,
            ele=15,
            alph=1,
            bg=1,
            colmap=False,
            cmap_type=cm.viridis,
            show_dists=False,
            vmin_min=-1,
            vmax_max=1):
    """
    creates figure and plots a point cloud in a chosen color and density 

    input: pA - 3xn
    """
    if np.shape(pA)[0] != 3:
        sys.exit('use 3xn point arrays as input')
    xW, yW, zW = pA
    # plot a sparse version of the xyz-data
    x = np.float32(xW[0::sparse])
    y = np.float32(yW[0::sparse])
    z = np.float32(zW[0::sparse])
    z_mean = np.mean(z)
    z_std = np.std(z)
    fig = plt.figure()  # plt.gcf()
    fig.suptitle('Reconstructed Point Cloud (in px)')
    rc('font', size=12)
    rc('font', family='Segoe UI')  # serif
    rc('axes', labelsize=12)
    fig.set_size_inches(8, 8)

    fig.patch.set_facecolor('white')
    ax1 = fig.add_subplot(111, projection='3d')  # = Axes3D(fig)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    if colmap == True:
        ax1.scatter(x, y, z, s=ms, c=color, vmin=z_mean - 2*z_std,
                    vmax=z_mean + 2*z_std, depthshade=ds, lw=0, alpha=alph, cmap=cmap_type)
    elif show_dists == True:
        ax1.scatter(x, y, z, s=ms, c=color, vmin=vmin_min,
                    vmax=vmax_max, depthshade=ds, lw=0, alpha=alph, cmap=cmap_type)
    else:
        ax1.scatter(x, y, z, s=ms, c=color, vmin=z_mean - 2*z_std,
                    vmax=z_mean + 2*z_std, depthshade=ds, lw=0, alpha=alph)

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array(
        [x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
    xb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -
                                1:2:2][0].flatten() + 0.5*(x.max()+x.min())
    yb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -
                                1:2:2][1].flatten() + 0.5*(y.max()+y.min())
    zb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -
                                1:2:2][2].flatten() + 0.5*(z.max()+z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(xb, yb, zb):
        ax1.plot([xb], [yb], [zb], 'w')

    ax1.grid(False)
    ax1.set_axis_off()

    if bg == 0:
        # get rid of colored axes planes
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False

        # set color to white (or whatever is "invisible")
        ax1.xaxis.pane.set_edgecolor('w')
        ax1.yaxis.pane.set_edgecolor('w')
        ax1.zaxis.pane.set_edgecolor('w')

    ax1.azim = azi
    ax1.elev = ele
    plt.show()


def custom_draw_geometry(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 0.7
    opt.light_on = False
    vis.run()
    vis.destroy_window()

# Copyright (c) 2023, Stefan Toeberg.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# (http://opensource.org/licenses/BSD-3-Clause)
#
# __author__ = "Stefan Toeberg, LUH: IMR"


def circles(x, y, s, c="b", vmin=None, vmax=None, **kwargs):
    """
    Make a scatter of circles plot of x vs y, where x and y are sequence
    like objects of the same lengths. The size of circles are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circle in data unit.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """

    if np.isscalar(c):
        kwargs.setdefault("color", c)
        c = None
    if "fc" in kwargs:
        kwargs.setdefault("facecolor", kwargs.pop("fc"))
    if "ec" in kwargs:
        kwargs.setdefault("edgecolor", kwargs.pop("ec"))
    if "ls" in kwargs:
        kwargs.setdefault("linestyle", kwargs.pop("ls"))
    if "lw" in kwargs:
        kwargs.setdefault("linewidth", kwargs.pop("lw"))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
    return collection
