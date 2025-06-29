import os
from pathlib import Path
import subprocess
import argparse

import numpy as np
import nibabel as nib
import nibabel.processing


def get_bbox_from_mask(mask, outside_value=-900, addon=0):
    if type(addon) is int:
        addon = [addon] * 3
    if (mask > outside_value).sum() == 0:
        print("WARNING: Could not crop because no foreground detected")
        minzidx, maxzidx = 0, mask.shape[0]
        minxidx, maxxidx = 0, mask.shape[1]
        minyidx, maxyidx = 0, mask.shape[2]
    else:
        mask_voxel_coords = np.where(mask > outside_value)
        minzidx = int(np.min(mask_voxel_coords[0])) - addon[0]
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1 + addon[0]
        minxidx = int(np.min(mask_voxel_coords[1])) - addon[1]
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1 + addon[1]
        minyidx = int(np.min(mask_voxel_coords[2])) - addon[2]
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1 + addon[2]

    # Avoid bbox to get out of image size
    s = mask.shape
    minzidx = max(0, minzidx)
    maxzidx = min(s[0], maxzidx)
    minxidx = max(0, minxidx)
    maxxidx = min(s[1], maxxidx)
    minyidx = max(0, minyidx)
    maxyidx = min(s[2], maxyidx)

    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    """
    image: 3d ndarray
    bbox: list of lists [[minx_idx, maxx_idx], [miny_idx, maxy_idx], [minz_idx, maxz_idx]]
          Indices of bbox must be in voxel coordinates  (not in world space)
    """
    assert len(image.shape) == 3, "only supports 3d images"
    return image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]


def crop_to_bbox_nifti(image: nib.Nifti1Image, bbox, dtype=None) -> nib.Nifti1Image:
    """
    Crop nifti image to bounding box and adapt affine accordingly

    image: nib.Nifti1Image
    bbox: list of lists [[minx_idx, maxx_idx], [miny_idx, maxy_idx], [minz_idx, maxz_idx]]
          Indices of bbox must be in voxel coordinates  (not in world space)
    dtype: dtype of the output image

    returns: nib.Nifti1Image
    """
    assert len(image.shape) == 3, "only supports 3d images"
    data = image.get_fdata()

    # Crop the image
    data_cropped = crop_to_bbox(data, bbox)

    # Update the affine matrix
    affine = np.copy(image.affine)
    affine[:3, 3] = np.dot(affine, np.array([bbox[0][0], bbox[1][0], bbox[2][0], 1]))[:3]

    data_type = image.dataobj.dtype if dtype is None else dtype
    return nib.Nifti1Image(data_cropped.astype(data_type), affine)


def crop_to_mask(img_in, mask_img, addon=[0,0,0], dtype=None, verbose=False):
    """
    Crops a nifti image to a mask and adapts the affine accordingly.

    img_in: nifti image
    mask_img: nifti image
    addon = addon in mm along each axis
    dtype: output dtype

    Returns a nifti image.
    """
    # This is needed for body mask with sometimes does not have the same shape as the
    # input image because it was generated on a lower resolution.
    # (normally the body mask should be resampled to the original resolution, but it
    # might have been generated by a different program)
    # This is quite slow for large images. Since normally not needed we remove it.
    #
    # print("Transforming crop mask to img space:")
    # print(f"  before: {mask_img.shape}")
    # mask_img = nibabel.processing.resample_from_to(mask_img, img_in, order=0)
    # print(f"  after: {mask_img.shape}")

    mask = mask_img.get_fdata()

    addon = (np.array(addon) / img_in.header.get_zooms()).astype(int)  # mm to voxels
    bbox = get_bbox_from_mask(mask, outside_value=0, addon=addon)

    img_out = crop_to_bbox_nifti(img_in, bbox, dtype)
    return img_out, bbox


def crop_to_mask_nifti(img_path, mask_path, out_path, addon=[0,0,0], dtype=None, verbose=False):
    """
    Crops a nifti image to a mask and adapts the affine accordingly.

    img_path: nifti image path
    mask_path: nifti image path
    out_path: output path
    addon = addon in mm along each axis
    dtype: output dtype

    Returns bbox coordinates.
    """
    img_in = nib.load(img_path)
    mask_img = nib.load(mask_path)

    img_out, bbox = crop_to_mask(img_in, mask_img, addon, dtype, verbose)

    nib.save(img_out, out_path)
    return bbox


def undo_crop(img, ref_img, bbox):
    """
    Fit the image which was cropped by bbox back into the shape of ref_img.
    """
    if img.ndim == 3:
        img_out = np.zeros(ref_img.shape, dtype=img.get_data_dtype())
        img_out[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = img.get_fdata()
    elif img.ndim == 4:
        img_out = np.zeros(ref_img.shape + (img.shape[3],), dtype=img.get_data_dtype())
        img_out[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1], :] = img.get_fdata()
    return nib.Nifti1Image(img_out, ref_img.affine)

def undo_crop_probabilities(prob, ref_img, bbox):
    prob_out = np.zeros(ref_img.shape + (prob.shape[3],))
    prob_out[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1], :] = prob.get_fdata()
    prob.uncache()
    return nib.Nifti1Image(prob_out, ref_img.affine)

def undo_crop_nifti(img_path, ref_img_path, bbox, out_path):
    """
    Fit the image which was cropped by bbox back into the shape of ref_img.
    """
    img = nib.load(img_path)
    ref_img = nib.load(ref_img_path)

    img_out = undo_crop(img, ref_img, bbox)

    nib.save(img_out, out_path)
