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
#                   creation of the initialization mask to perform GrabCut
#                   and sequential processing of the masking of disparity maps
#                   """


# libraries
import numpy as np
from PIL import Image
import cv2
from skimage import morphology
import copy
from matplotlib import pyplot as plt
from matplotlib import rc


def grab_cut_refinement(imgMask, img1, segmentResizedImage, thickness):
    # performed iterations of GrabCut, 3 is generally sufficient
    numIters = 3

    if segmentResizedImage == True:
        # perform GrabCut based on the resized image which was used in the UNet segmentation
        imgShape = np.shape(imgMask)
        img1Resized = img1.resize(imgShape, resample=Image.BICUBIC)
        # define the widths of the probable fore- and background
        maskWidth = convert_number_to_width(thickness, np.shape(img1Resized)[0])
    else:
        # use original image and rescale the mask later
        imgShape = np.shape(img1)
        # define the widths of the probable fore- and background
        maskWidth = convert_number_to_width(thickness, imgShape[0])

    # resize mask output of the Unet to original imageSize
    maskImgSize = imgMask.resize(imgShape, resample=Image.BICUBIC)
    maskImgSizeBin = get_binary_mask(np.asarray(maskImgSize), thresh=20)
    maskImgSizeBool = maskImgSizeBin < 10

    # improve mask (remove small objects) and fill contour
    maskClean = morphology.remove_small_objects(
        maskImgSizeBool, min_size=imgShape[0] * 2
    )
    contour, posContour, imageContour = get_contour(
        maskClean.astype(np.uint8), threshold=0.5
    )

    # create a single channel black image
    imgBlack = np.zeros(imgShape)
    maskFilled = cv2.fillPoly(
        imgBlack, pts=[contour[posContour]], color=(255, 255, 255)
    )

    # expand and narrow contour
    maskOuter, maskInner, contInitial, contOuter, contInner = expand_contour(
        maskFilled.astype(np.uint8), maskWidth
    )
    maskGrabcut = copy.copy(maskOuter)
    outerRegion = maskFilled + maskOuter
    innerRegion = maskFilled + maskInner
    maskGrabcut[maskOuter == 255] = 1
    maskGrabcut[outerRegion == 255] = 2
    maskGrabcut[innerRegion == 255] = 3

    # get inputs of cv2.grabcut
    if segmentResizedImage == True:
        imgTmp, bgdmodel, fgdmodel, rect = get_input_GC(img1Resized)
    else:
        imgTmp, bgdmodel, fgdmodel, rect = get_input_GC(img1)

    # perform grabcut
    output = cv2.grabCut(
        imgTmp, maskGrabcut, rect, bgdmodel, fgdmodel, numIters, cv2.GC_INIT_WITH_MASK
    )

    # manipulate output to obtain correctly sized mask
    maskNew = copy.deepcopy(output[0])
    maskFinal = np.where((maskNew == 1) + (maskNew == 3), 255, 0).astype("uint8")

    if segmentResizedImage == True:
        maskFinalResized = recover_original_mask_size(maskFinal, img1)
    else:
        maskFinalResized = maskFinal

    return maskFinalResized, maskGrabcut


def recover_original_mask_size(maskFinal, img1):
    maskResized = cv2.resize(
        maskFinal, dsize=np.shape(img1)[0:2], interpolation=cv2.INTER_CUBIC
    )
    maskBin = get_binary_mask(maskResized, thresh=20)
    maskBool = maskBin < 10
    # improve mask (remove small objects) and fill contour
    maskClean = morphology.remove_small_objects(
        maskBool, min_size=np.shape(img1)[0] * 2
    )
    finalMaskResized = maskClean.astype(np.uint8) * 255
    finalMaskResized[finalMaskResized < 10] = 1
    finalMaskResized[finalMaskResized > 10] = 0
    finalMaskResized = finalMaskResized * 255
    return finalMaskResized


def convert_number_to_width(number, imgWidth):
    """
    converts an integer number into the absolute width in pixels
    and other numbers into the percentage image width
    """

    if type(number) is not int:
        maskWidth = int(number * imgWidth)
    else:
        maskWidth = number

    return maskWidth


def get_binary_mask(mask, thresh):
    mask_bin = copy.deepcopy(mask)
    mask_bin[mask < thresh] = 0
    mask_bin[mask >= thresh] = 255
    mask_bin = np.uint8(mask_bin)

    return mask_bin


def get_input_GC(img):
    img1 = copy.deepcopy(img)
    img1 = np.uint8(img1)
    img1 = np.stack((img1,) * 3, axis=-1)
    img_new = cv2.resize(
        img1, (int(np.shape(img1)[1] - 0), int(np.shape(img1)[0] - 0))
    )
    img2 = copy.deepcopy(img_new)
    img2 = np.uint8(img2)

    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)
    rect = (0, 0, 1, 1)

    return img2, bgdmodel, fgdmodel, rect


def get_contour(mask, threshold=127):
    ret, thresh_ct = cv2.threshold(mask, threshold, 255, 0)
    contours, a = cv2.findContours(thresh_ct, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
    pos_ct = get_position_contour(contours)

    shape = np.shape(mask)
    img_ct = np.zeros((shape[0], shape[1]))
    length = len(contours[pos_ct])
    for k in range(length):
        img_ct[contours[pos_ct][k][0][1], contours[pos_ct][k][0][0]] = 255

    return contours, pos_ct, img_ct


def get_position_contour(contour_list):
    length = len(contour_list)
    array = np.zeros((length, 2))
    for i in range(length):
        array[i, 0] = i
        array[i, 1] = np.shape(contour_list[i])[0]
    sorted_array = array[array[:, 1].argsort()[::-1]]

    position = sorted_array[0, 0]
    position = np.uint8(position)

    return position


def expand_contour(mask, iterations_exp=5):
    # dilate mask with 3x3 kernel and find the new contour
    kernel = np.ones((3, 3), np.uint8)
    kernel[0, 0] = kernel[0, 2] = kernel[2, 0] = kernel[2, 0] = kernel[2, 2] = 0

    dilated_masks, eroded_masks = [], []
    dilated_masks.append(mask)
    eroded_masks.append(mask)
    contour_dilated, contour_eroded = [], []

    ret_0, thresh_0 = cv2.threshold(mask, 127, 255, 0)
    contour_0 = cv2.findContours(
        thresh_0.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )[-2:]

    i = 0
    while i < iterations_exp:
        mask_dil = copy.deepcopy(dilated_masks[i])
        mask_erod = copy.deepcopy(eroded_masks[i])
        mask_dilated = cv2.dilate(mask_dil, kernel, iterations=1)
        mask_eroded = cv2.erode(mask_erod, kernel, iterations=1)
        ret_dil, thresh_dil = cv2.threshold(mask_dilated, 127, 255, 0)
        ret_erod, thresh_erod = cv2.threshold(mask_eroded, 127, 255, 0)
        contour_dil, a = cv2.findContours(
            thresh_dil, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )[-2:]
        contour_erod, b = cv2.findContours(
            thresh_erod, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )[-2:]
        contour_dilated.append(contour_dil)
        contour_eroded.append(contour_erod)

        # find values for the contour pixels
        # (random int in the range [minval, maxval] of the window)
        for j in range(len(contour_dil[0])):
            indices_ctr = contour_dil[0][j]
            s = indices_ctr[0][0]
            z = indices_ctr[0][1]
            a_value = 255
            mask_dilated[z, s] = a_value

        # find gaps
        mask_dilated_bool = mask_dilated > 0
        image_bool = mask_dilated < 1

        erg = mask_dilated_bool & image_bool
        gaps = np.where(erg == True)

        for l in range(len(gaps[0])):
            z = gaps[0][l]
            s = gaps[1][l]
            a_value = 255
            mask_dilated[z, s] = a_value

        dilated_masks.append(mask_dilated)
        eroded_masks.append(mask_eroded)

        i += 1

    # draw contour_start
    shape = np.shape(dilated_masks[iterations_exp])
    contour_start = np.zeros((shape[0], shape[1]))
    length = len(contour_0[0][0])  # [0][0]
    for k in range(length):
        contour_start[
            contour_0[0][0][k][0][1], contour_0[0][0][k][0][0]
        ] = 255  # [0][0]

    # draw contour_end_big
    shape = np.shape(dilated_masks[iterations_exp])
    contour_end_big = np.zeros((shape[0], shape[1]))
    length = len(contour_dilated[iterations_exp - 1][0])
    for k in range(length):
        contour_end_big[
            contour_dilated[iterations_exp - 1][0][k][0][1],
            contour_dilated[iterations_exp - 1][0][k][0][0],
        ] = 255

    # draw contour_end_small
    shape = np.shape(dilated_masks[iterations_exp])
    contour_end_small = np.zeros((shape[0], shape[1]))
    length = len(contour_eroded[iterations_exp - 1][0])
    for k in range(length):
        contour_end_small[
            contour_eroded[iterations_exp - 1][0][k][0][1],
            contour_eroded[iterations_exp - 1][0][k][0][0],
        ] = 255

    return (
        dilated_masks[iterations_exp],
        eroded_masks[iterations_exp],
        contour_start,
        contour_end_big,
        contour_end_small,
    )


def mask_disp(disp, specimen_mask, padded_zeroes=300, plot=True):
    """
    gets the disparity map of the specimen only and sets all other values
    depending on the given mask to invalid
    inputs:      disp              - nxm disparity map
                 maskParticle      - nxm mask - pixels of the particle are 1s, rest 0s
                 paddedZ           - number of padded zeros
    """

    # get the minimum disparity value for invalid pixels
    disp_invalid = np.abs(np.min(disp))

    # remove area of padded zeros on the sides of the image
    if padded_zeroes != 0:
        disp_cropped = disp[padded_zeroes:-padded_zeroes, padded_zeroes:-padded_zeroes]
    else:
        disp_cropped = disp
    # create mask for the actual mask, forcing 1s and 0s
    new_mask = np.isin(specimen_mask, 0)

    # mask the disp with the correct mask, setting all invalid pixels (in mask
    # zeros) to -disp_invalid
    disp_masked = (np.float32(disp_cropped) + np.float32(disp_invalid)) * np.float32(
        ~new_mask
    ) - disp_invalid

    # plot the masked disparity map
    if plot == True:
        # display the disparity map after the particle segmentation
        plt.figure()
        plt.imshow(disp_masked, "gray")
        plt.title("Disparity Map by SGBM in px")
        cbar = plt.colorbar()
        cbar.solids.set_edgecolor("face")
        rc("font", size=12)
        rc("font", family="Segoe UI")
        plt.draw()
        plt.show()
    return disp_masked
