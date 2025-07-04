import os
import sys
import json
import random
import string
import pickle
import time
import platform
import shutil
import subprocess
from pathlib import Path
from os.path import join
from typing import Union
from functools import partial
from multiprocessing import Pool
import tempfile
import inspect
import warnings

import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from p_tqdm import p_map
import torch

from totalsegmentator.libs import nostdout

# nnUNet 2.1
# with nostdout():
#     from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data
# nnUNet 2.2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from nnunetv2.utilities.file_path_utilities import get_output_folder

from totalsegmentator.map_to_binary import class_map, class_map_5_parts, class_map_parts_mr, class_map_parts_headneck_muscles
from totalsegmentator.map_to_binary import map_taskid_to_partname_mr, map_taskid_to_partname_ct, map_taskid_to_partname_headneck_muscles
from totalsegmentator.alignment import as_closest_canonical_nifti, undo_canonical_nifti
from totalsegmentator.alignment import as_closest_canonical, undo_canonical
from totalsegmentator.resampling import change_spacing
from totalsegmentator.libs import combine_masks, compress_nifti, check_if_shape_and_affine_identical, reorder_multilabel_like_v1
from totalsegmentator.dicom_io import dcm_to_nifti, save_mask_as_rtstruct
from totalsegmentator.cropping import crop_to_mask_nifti, undo_crop_nifti
from totalsegmentator.cropping import crop_to_mask, undo_crop, undo_crop_probabilities
from totalsegmentator.postprocessing import remove_outside_of_mask, extract_skin, remove_auxiliary_labels
from totalsegmentator.postprocessing import keep_largest_blob_multilabel, remove_small_blobs_multilabel
from totalsegmentator.nifti_ext_header import save_multilabel_nifti, add_label_map_to_nifti
from totalsegmentator.statistics import get_basic_statistics

# Test time augmentation
from uncertainty.aleatoric import test_time_augmentation

# Hide nnunetv2 warning: Detected old nnU-Net plans format. Attempting to reconstruct network architecture...
warnings.filterwarnings("ignore", category=UserWarning, module="nnunetv2")
warnings.filterwarnings("ignore", category=FutureWarning, module="nnunetv2")  # ignore torch.load warning


def _get_full_task_name(task_id: int, src: str="raw"):
    if src == "raw":
        base = Path(os.environ['nnUNet_raw_data_base']) / "nnUNet_raw_data"
    elif src == "preprocessed":
        base = Path(os.environ['nnUNet_preprocessed'])
    elif src == "results":
        base = Path(os.environ['RESULTS_FOLDER']) / "nnUNet" / "3d_fullres"
    dirs = [str(dir).split("/")[-1] for dir in base.glob("*")]
    for dir in dirs:
        if f"Task{task_id:03d}" in dir:
            return dir

    # If not found in 3d_fullres, search in 3d_lowres
    if src == "results":
        base = Path(os.environ['RESULTS_FOLDER']) / "nnUNet" / "3d_lowres"
        dirs = [str(dir).split("/")[-1] for dir in base.glob("*")]
        for dir in dirs:
            if f"Task{task_id:03d}" in dir:
                return dir

    # If not found in 3d_lowres, search in 2d
    if src == "results":
        base = Path(os.environ['RESULTS_FOLDER']) / "nnUNet" / "2d"
        dirs = [str(dir).split("/")[-1] for dir in base.glob("*")]
        for dir in dirs:
            if f"Task{task_id:03d}" in dir:
                return dir

    raise ValueError(f"task_id {task_id} not found")


def contains_empty_img(imgs):
    """
    imgs: List of image paths
    """
    is_empty = True
    for img in imgs:
        this_is_empty = len(np.unique(nib.load(img).get_fdata())) == 1
        is_empty = is_empty and this_is_empty
    return is_empty


def supports_keyword_argument(func, keyword: str):
    """
    Check if a function supports a specific keyword argument.

    Returns:
    - True if the function supports the specified keyword argument.
    - False otherwise.
    """
    signature = inspect.signature(func)
    parameters = signature.parameters
    return keyword in parameters



def nnUNet_predict(dir_in, dir_out, task_id, model="3d_fullres", folds=None,
                   trainer="nnUNetTrainerV2", tta=False,
                   num_threads_preprocessing=6, num_threads_nifti_save=2):
    """
    Identical to bash function nnUNet_predict

    folds:  folds to use for prediction. Default is None which means that folds will be detected
            automatically in the model output folder.
            for all folds: None
            for only fold 0: [0]
    """
    with nostdout():
        from nnunet.inference.predict import predict_from_folder
        from nnunet.paths import default_plans_identifier, network_training_output_dir, default_trainer

    save_npz = False
    # num_threads_preprocessing = 6
    # num_threads_nifti_save = 2
    # num_threads_preprocessing = 1
    # num_threads_nifti_save = 1
    lowres_segmentations = None
    part_id = 0
    num_parts = 1
    disable_tta = not tta
    overwrite_existing = False
    mode = "normal" if model == "2d" else "fastest"
    all_in_gpu = None
    step_size = 0.5
    chk = "model_final_checkpoint"
    disable_mixed_precision = False

    task_id = int(task_id)
    task_name = _get_full_task_name(task_id, src="results")

    # trainer_class_name = default_trainer
    # trainer = trainer_class_name
    plans_identifier = default_plans_identifier

    model_folder_name = join(network_training_output_dir, model, task_name, trainer + "__" + plans_identifier)
    print("using model stored in ", model_folder_name)

    predict_from_folder(model_folder_name, dir_in, dir_out, folds, save_npz, num_threads_preprocessing,
                        num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                        overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                        mixed_precision=not disable_mixed_precision,
                        step_size=step_size, checkpoint_name=chk)


def nnUNetv2_predict(dir_in, dir_out, task_id, model="3d_fullres", folds=None,
                     trainer="nnUNetTrainer", tta=False,
                     num_threads_preprocessing=3, num_threads_nifti_save=2,
                     plans="nnUNetPlans", device="cuda", quiet=False, step_size=0.5, save_probs=False):
    """
    Identical to bash function nnUNetv2_predict

    folds:  folds to use for prediction. Default is None which means that folds will be detected
            automatically in the model output folder.
            for all folds: None
            for only fold 0: [0]
    """
    dir_in = str(dir_in)
    dir_out = str(dir_out)

    model_folder = get_output_folder(task_id, trainer, plans, model)

    assert device in ['cpu', 'cuda',
                           'mps'] or isinstance(device, torch.device), f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {device}.'
    if device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        # torch.set_num_interop_threads(1)  # throws error if setting the second time
        device = torch.device('cuda')
    elif isinstance(device, torch.device):
        torch.set_num_threads(1)
        device = device
    else:
        device = torch.device('mps')
    disable_tta = not tta
    verbose = False
    save_probabilities = save_probs
    continue_prediction = False
    chk = "checkpoint_final.pth"
    npp = num_threads_preprocessing
    nps = num_threads_nifti_save
    prev_stage_predictions = None
    num_parts = 1
    part_id = 0
    allow_tqdm = not quiet

    # nnUNet 2.1
    # predict_from_raw_data(dir_in,
    #                       dir_out,
    #                       model_folder,
    #                       folds,
    #                       step_size,
    #                       use_gaussian=True,
    #                       use_mirroring=not disable_tta,
    #                       perform_everything_on_gpu=True,
    #                       verbose=verbose,
    #                       save_probabilities=save_probabilities,
    #                       overwrite=not continue_prediction,
    #                       checkpoint_name=chk,
    #                       num_processes_preprocessing=npp,
    #                       num_processes_segmentation_export=nps,
    #                       folder_with_segs_from_prev_stage=prev_stage_predictions,
    #                       num_parts=num_parts,
    #                       part_id=part_id,
    #                       device=device)

    # nnUNet 2.2.1
    if supports_keyword_argument(nnUNetPredictor, "perform_everything_on_gpu"):
        predictor = nnUNetPredictor(
            tile_step_size=step_size,
            use_gaussian=True,
            use_mirroring=not disable_tta,
            perform_everything_on_gpu=True,  # for nnunetv2<=2.2.1
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose,
            allow_tqdm=allow_tqdm
        )
    # nnUNet >= 2.2.2
    else:
        predictor = nnUNetPredictor(
            tile_step_size=step_size,
            use_gaussian=True,
            use_mirroring=not disable_tta,
            perform_everything_on_device=True,  # for nnunetv2>=2.2.2
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose,
            allow_tqdm=allow_tqdm
        )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=folds,
        checkpoint_name=chk,
    )
    # new nnunetv2 feature: keep dir_out empty to return predictions as return value
    predictor.predict_from_files(dir_in, dir_out,
                                 save_probabilities=save_probabilities, overwrite=not continue_prediction,
                                 num_processes_preprocessing=npp, num_processes_segmentation_export=nps,
                                 folder_with_segs_from_prev_stage=prev_stage_predictions,
                                 num_parts=num_parts, part_id=part_id)

    # # Use numpy as input. TODO: In entire pipeline do not save to disk
    # input_image = nib.load(Path(dir_in) / "s01_0000.nii.gz")
    # input_data = np.asanyarray(input_image.dataobj).transpose(2, 1, 0)[None,...].astype(np.float32)
    # spacing = input_image.header.get_zooms()
    # affine = input_image.affine
    # # Do i have to transpose spacing? does not matter because anyways isotropic at this point.
    # spacing = (spacing[2], spacing[1], spacing[0])
    # props = {"spacing": spacing}
    # # from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    # # input_data, props = SimpleITKIO().read_images([os.path.join(dir_in, "s01_0000.nii.gz")])
    # seg = predictor.predict_single_npy_array(input_data, props,
    #                                          prev_stage_predictions, None,
    #                                          save_probabilities)
    # seg = seg.transpose(2, 1, 0)
    # nib.save(nib.Nifti1Image(seg.astype(np.uint8), affine), Path(dir_out) / "s01.nii.gz")
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def save_segmentation_nifti(class_map_item, tmp_dir=None, file_out=None, nora_tag=None, header=None, task_name=None, quiet=None):
    k, v = class_map_item
    # Have to load img inside of each thread. If passing it as argument a lot slower.
    if not task_name.startswith("total") and not quiet:
        print(f"Creating {v}.nii.gz")
    img = nib.load(tmp_dir / "s01.nii.gz")
    img_data = img.get_fdata()
    binary_img = img_data == k
    output_path = str(file_out / f"{v}.nii.gz")
    nib.save(nib.Nifti1Image(binary_img.astype(np.uint8), img.affine, header), output_path)
    if nora_tag != "None":
        subprocess.call(f"/opt/nora/src/node/nora -p {nora_tag} --add {output_path} --addtag mask", shell=True)

def nnUNet_predict_image(file_in: Union[str, Path, Nifti1Image], file_out, task_id, model="3d_fullres", folds=None,
                         trainer="nnUNetTrainerV2", tta=False, multilabel_image=True,
                         resample=None, crop=None, crop_path=None, task_name="total", nora_tag="None", preview=False,
                         save_binary=False, nr_threads_resampling=1, nr_threads_saving=6, force_split=False,
                         crop_addon=[3,3,3], roi_subset=None, output_type="nifti",
                         statistics=False, quiet=False, verbose=False, test=0, skip_saving=False,
                         device="cuda", exclude_masks_at_border=True, no_derived_masks=False,
                         v1_order=False, stats_aggregation="mean", remove_small_blobs=False, save_probs=False, augment=False):
    """
    crop: string or a nibabel image
    resample: None or float (target spacing for all dimensions) or list of floats
    """
    if not isinstance(file_in, Nifti1Image):
        file_in = Path(file_in)
        if str(file_in).endswith(".nii") or str(file_in).endswith(".nii.gz"):
            img_type = "nifti"
        else:
            img_type = "dicom"
        if not file_in.exists():
            sys.exit("ERROR: The input file or directory does not exist.")
    else:
        img_type = "nifti"
    if file_out is not None:
        file_out = Path(file_out)
    multimodel = type(task_id) is list

    if img_type == "nifti" and output_type == "dicom":
        raise ValueError("To use output type dicom you also have to use a Dicom image as input.")

    if task_name == "total":
        class_map_parts = class_map_5_parts
        map_taskid_to_partname = map_taskid_to_partname_ct
    elif task_name == "total_mr":
        class_map_parts = class_map_parts_mr
        map_taskid_to_partname = map_taskid_to_partname_mr
    elif task_name == "headneck_muscles":
        class_map_parts = class_map_parts_headneck_muscles
        map_taskid_to_partname = map_taskid_to_partname_headneck_muscles
    
    if type(resample) is float:
        resample = [resample, resample, resample]
    
    if v1_order and task_name == "total":
        label_map = class_map["total_v1"]
    else:
        label_map = class_map[task_name]
    
    # Keep only voxel values corresponding to the roi_subset
    if roi_subset is not None:
        label_map = {k: v for k, v in label_map.items() if v in roi_subset}
            
    # for debugging
    # tmp_dir = file_in.parent / ("nnunet_tmp_" + ''.join(random.Random().choices(string.ascii_uppercase + string.digits, k=8)))
    # (tmp_dir).mkdir(exist_ok=True)
    # with tmp_dir as tmp_folder:
    with tempfile.TemporaryDirectory(prefix="nnunet_tmp_") as tmp_folder:
        tmp_dir = Path(tmp_folder)
        if verbose: print(f"tmp_dir: {tmp_dir}")

        if img_type == "dicom":
            if not quiet: print("Converting dicom to nifti...")
            (tmp_dir / "dcm").mkdir()  # make subdir otherwise this file would be included by nnUNet_predict
            dcm_to_nifti(file_in, tmp_dir / "dcm" / "converted_dcm.nii.gz", tmp_dir, verbose=verbose)
            file_in_dcm = file_in
            file_in = tmp_dir / "dcm" / "converted_dcm.nii.gz"
            
            # for debugging
            # shutil.copy(file_in, file_in_dcm.parent / "converted_dcm_TMP.nii.gz")

            # Workaround to be able to access file_in on windows (see issue #106)
            # if platform.system() == "Windows":
            #     file_in = file_in.NamedTemporaryFile(delete = False)
            #     file_in.close()

            # if not multilabel_image:
            #     shutil.copy(file_in, file_out / "input_file.nii.gz")
            if not quiet: print(f"  found image with shape {nib.load(file_in).shape}")

        if isinstance(file_in, Nifti1Image):
            img_in_orig = file_in
        else:
            img_in_orig = nib.load(file_in)
        if len(img_in_orig.shape) == 2:
            raise ValueError("TotalSegmentator does not work for 2D images. Use a 3D image.")
        if len(img_in_orig.shape) > 3:
            print(f"WARNING: Input image has {len(img_in_orig.shape)} dimensions. Only using first three dimensions.")
            img_in_orig = nib.Nifti1Image(img_in_orig.get_fdata()[:,:,:,0], img_in_orig.affine)
            
        img_dtype = img_in_orig.get_data_dtype()
        if img_dtype.fields is not None:
            raise TypeError(f"Invalid dtype {img_dtype}. Expected a simple dtype, not a structured one.")

        # takes ~0.9s for medium image
        img_in = nib.Nifti1Image(img_in_orig.get_fdata(), img_in_orig.affine)  # copy img_in_orig

        if crop is not None:
            if type(crop) is str:
                if crop == "lung" or crop == "pelvis":
                    crop_mask_img = combine_masks(crop_path, crop)
                else:
                    crop_mask_img = nib.load(crop_path / f"{crop}.nii.gz")
            else:
                crop_mask_img = crop
                
            if crop_mask_img.get_fdata().sum() == 0:
                if not quiet: 
                    print("INFO: Crop is empty. Returning empty segmentation.")
                img_out = nib.Nifti1Image(np.zeros(img_in.shape, dtype=np.uint8), img_in.affine)
                img_out = add_label_map_to_nifti(img_out, label_map)
                if file_out is not None:
                    nib.save(img_out, file_out)
                if nora_tag != "None":
                    subprocess.call(f"/opt/nora/src/node/nora -p {nora_tag} --add {file_out} --addtag atlas", shell=True)
                return img_out, img_in_orig, None
                
            img_in, bbox = crop_to_mask(img_in, crop_mask_img, addon=crop_addon, dtype=np.int32,
                                      verbose=verbose)
            if not quiet:
                print(f"  cropping from {crop_mask_img.shape} to {img_in.shape}")

        img_in = as_closest_canonical(img_in)

        if resample is not None:
            if not quiet: print("Resampling...")
            st = time.time()
            img_in_shape = img_in.shape
            img_in_zooms = img_in.header.get_zooms()
            img_in_rsp = change_spacing(img_in, resample,
                                        order=3, dtype=np.int32, nr_cpus=nr_threads_resampling)  # 4 cpus instead of 1 makes it a bit slower
            if verbose:
                print(f"  from shape {img_in.shape} to shape {img_in_rsp.shape}")
            if not quiet: print(f"  Resampled in {time.time() - st:.2f}s")
        else:
            img_in_rsp = img_in


        if augment:
            augmenter = test_time_augmentation(img_in_rsp)
            augs = augmenter.do_augment(do_rotation=True, do_gauss_noise=True, do_gauss_blur=True,
                   do_brightness=True, do_contrast=True, do_low_res=True, do_gamma=True,
                   do_mirror=True)
            aug_probs = {}
            for k, v in augs.items():
                (tmp_dir / k).mkdir(exist_ok=True)
                nib.save(v, tmp_dir / k / "s01_0000.nii.gz")

            # todo important: change
            nr_voxels_thr = 512 * 512 * 900
            # nr_voxels_thr = 256*256*900
            img_parts = ["s01"]
            ss = img_in_rsp.shape
            # If image to big then split into 3 parts along z axis. Also make sure that z-axis is at least 200px otherwise
            # splitting along it does not really make sense.
            do_triple_split = np.prod(ss) > nr_voxels_thr and ss[2] > 200 and multimodel
            if force_split:
                do_triple_split = True

            if do_triple_split:
                raise Exception('Not currently supported for test time augmentation')

            if task_name == "total" and resample is not None and resample[0] < 3.0:
                # overall speedup for 15mm model roughly 11% (GPU) and 100% (CPU)
                # overall speedup for  3mm model roughly  0% (GPU) and  10% (CPU)
                # (dice 0.001 worse on test set -> ok)
                # (for lung_trachea_bronchia somehow a lot lower dice)
                step_size = 0.8
            else:
                step_size = 0.5

            st = time.time()
            if multimodel:  # if running multiple models

                # only compute model parts containing the roi subset
                if roi_subset is not None:
                    part_names = []
                    new_task_id = []
                    for part_name, part_map in class_map_parts.items():
                        if any(organ in roi_subset for organ in part_map.values()):
                            # get taskid associated to model part_name
                            map_partname_to_taskid = {v: k for k, v in map_taskid_to_partname.items()}
                            new_task_id.append(map_partname_to_taskid[part_name])
                            part_names.append(part_name)
                    task_id = new_task_id
                    if save_probs:
                        selected_classes = class_map[task_name]
                        selected_classes_inv_new_map = dict(
                            zip([v for v in selected_classes.values() if v in roi_subset],
                                [i + 1 for i in range(len(roi_subset))]))

                    if verbose:
                        print(f"Computing parts: {part_names} based on the provided roi_subset")

                if test == 0:
                    class_map_inv = {v: k for k, v in class_map[task_name].items()}
                    for k in augs.keys():
                        #print(k)
                        (tmp_dir / k / "parts").mkdir(exist_ok=True)
                        seg_combined = {}
                        prob_combined = {}
                        # iterate over subparts of image
                        for img_part in img_parts:
                            img_shape = nib.load(tmp_dir / "original" / f"{img_part}_0000.nii.gz").shape
                        # Run several tasks and combine results into one segmentation
                        if roi_subset is not None:
                            num_classes = len(roi_subset)
                            prob_combined[img_part] = np.zeros(img_shape + (num_classes,), dtype=np.float32)
                        else:
                            num_classes = 0
                            for tmp_tid in task_id:
                                num_classes = num_classes + len(class_map_parts[map_taskid_to_partname[tmp_tid]])
                            prob_combined[img_part] = np.zeros(img_shape + (num_classes + 1,), dtype=np.float32)
                        for idx, tid in enumerate(task_id):
                            if not quiet: print(f"Predicting part {idx + 1} of {len(task_id)} ...")
                            with nostdout(verbose):
                                # nnUNet_predict(tmp_dir, tmp_dir, tid, model, folds, trainer, tta,
                                #                nr_threads_resampling, nr_threads_saving)
                                nnUNetv2_predict((tmp_dir / k), (tmp_dir / k), tid, model, folds, trainer, tta,
                                                 nr_threads_resampling, nr_threads_saving,
                                                 device=device, quiet=quiet, step_size=step_size, save_probs=save_probs)
                            # iterate over models (different sets of classes)
                            for img_part in img_parts:
                                if save_probs:
                                    (tmp_dir / k / f"{img_part}.npz").rename(tmp_dir / k / "parts" / f"{img_part}_{tid}.npz")
                                    prob = np.moveaxis(
                                        np.load(tmp_dir / k / "parts" / f"{img_part}_{tid}.npz")["probabilities"],
                                        [0, 1, 2, 3], [3, 2, 1, 0])
                                else:
                                    (tmp_dir / k / f"{img_part}.nii.gz").rename(
                                        tmp_dir / k /"parts" f"{img_part}_{tid}.nii.gz")
                                    seg = nib.load(tmp_dir / k / "parts" / f"{img_part}_{tid}.nii.gz").get_fdata()
                                if roi_subset is None:
                                    for jdx, class_name in class_map_parts[map_taskid_to_partname[tid]].items():
                                        prob_combined[img_part][:, :, :, class_map_inv[class_name]] = prob[:, :, :,jdx]

                                else:
                                    class_map_parts_sub = {k: v for k, v in
                                                           class_map_parts[map_taskid_to_partname[tid]].items() if
                                                           v in roi_subset}
                                    for jdx, class_name in class_map_parts_sub.items():
                                        prob_combined[img_part][:, :, :, selected_classes_inv_new_map[class_name] - 1] = prob[:, :, :, jdx]
                                del prob
                            aug_probs[k] = nib.Nifti1Image(prob_combined['s01'], img_in_rsp.affine)
            else:
                if not quiet: print("Predicting...")
                if test == 0:
                    with nostdout(verbose):
                        if save_probs:
                            if roi_subset is None:
                                num_classes = len(class_map_parts[map_taskid_to_partname[task_id]])
                            else:
                                num_classes = len(roi_subset)
                                selected_classes = class_map[task_name]
                                selected_classes_inv_new_map = dict(
                                    zip([v for v in selected_classes.values() if v in roi_subset],
                                        [i + 1 for i in range(len(roi_subset))]))
                            img_shape = img_in_rsp.shape
                        for k in augs.keys():
                            aug_prob = np.zeros(img_shape + (num_classes,), dtype=np.float32)
                            # nnUNet_predict(tmp_dir, tmp_dir, task_id, model, folds, trainer, tta,
                            #                nr_threads_resampling, nr_threads_saving)
                            nnUNetv2_predict((tmp_dir / k), (tmp_dir / k), task_id, model, folds, trainer, tta,
                                             nr_threads_resampling, nr_threads_saving,
                                             device=device, quiet=quiet, step_size=step_size, save_probs=save_probs)
                            if save_probs:
                                prob_pred = np.moveaxis(np.load(tmp_dir / k / f"{img_parts[0]}.npz")["probabilities"], [0, 1, 2, 3],[3, 2, 1, 0])
                                if roi_subset is None:
                                    class_map_inv = {v: k for k, v in class_map[task_name].items()}
                                    for jdx, class_name in class_map_parts[map_taskid_to_partname[task_id]].items():
                                        aug_prob[:, :, :, class_map_inv[class_name]] = prob_pred[:,:,:,jdx]
                                else:
                                    class_map_parts_sub = {k: v for k, v in
                                                           class_map_parts[map_taskid_to_partname[task_id]].items() if
                                                           v in roi_subset}
                                    for jdx, class_name in class_map_parts_sub.items():
                                        aug_prob[:, :, :, selected_classes_inv_new_map[class_name] - 1] = prob_pred[:, :, :, jdx]
                                del prob_pred
                                aug_probs[k] = nib.Nifti1Image(aug_prob, img_in_rsp.affine)
                # elif test == 2:
                #     print("WARNING: Using reference seg instead of prediction for testing.")
                #     shutil.copy(Path("tests") / "reference_files" / "example_seg_fast.nii.gz", tmp_dir / f"s01.nii.gz")
                elif test == 3:
                    print("WARNING: Using reference seg instead of prediction for testing.")
                    shutil.copy(Path("tests") / "reference_files" / "example_seg_lung_vessels.nii.gz",
                                tmp_dir / "s01.nii.gz")
            if not quiet: print(f"  Predicted in {time.time() - st:.2f}s")

            # Undo augmentations that alter image positioning
            probs = augmenter.undo_augment(aug_probs)
            seg, unc = augmenter.compute_entropy(probs)

            if resample is not None:
                seg = change_spacing(seg, resample, img_in_shape + (len(augs) + 1,), order=0, dtype=np.uint8, nr_cpus=6, force_affine=img_in.affine)
                unc = change_spacing(unc, resample, img_in_shape + (len(augs) + 1,), order=0, dtype=np.float32, nr_cpus=6, force_affine=img_in.affine)

            if crop is not None:
                seg = undo_crop(seg, img_in_orig, bbox)
                unc = undo_crop(unc, img_in_orig, bbox)

            # if resample is not None:
            #     for k,v in probs.items():
            #         probs[k] = change_spacing(v, resample, img_in_shape + (num_classes,),
            #                                        order=0, dtype=np.float32, nr_cpus=6,
            #                                        force_affine=img_in.affine)
            # if crop is not None:
            #     for k,v in probs.items():
            #         probs[k] = undo_crop_probabilities(v, img_in_orig, bbox)
            #
            # # Compute entropy across augmentations
            # seg, unc = augmenter.compute_entropy(probs)

            if file_out is not None and skip_saving is False:
                seg_header = img_in_orig.header.copy()
                unc_header = img_in_orig.header.copy()
                seg_header.set_data_dtype(np.uint8)
                unc_header.set_data_dtype(np.float32)
                name = str(file_in).split('\\')[-1].split('.')[0]
                if not quiet:
                    print(f"Saving segmentation for {name}")
                output_path_seg = str(file_out / f"{name}_seg.nii.gz")
                output_path_unc = str(file_out / f"{name}_unc.nii.gz")
                output_path_class = str(file_out / "classes.txt")
                nib.save(nib.Nifti1Image(seg.get_fdata(), seg.affine, seg_header), output_path_seg)
                nib.save(nib.Nifti1Image(unc.get_fdata(), unc.affine, unc_header), output_path_unc)
                if not os.path.exists(output_path_class):
                    filehandler = open(output_path_class, 'wt')
                    data = str(selected_classes_inv_new_map)
                    filehandler.write(data)

            return seg, unc, selected_classes_inv_new_map
        else:
            nib.save(img_in_rsp, tmp_dir / "s01_0000.nii.gz")


            # todo important: change
            nr_voxels_thr = 512*512*900
            # nr_voxels_thr = 256*256*900
            img_parts = ["s01"]
            ss = img_in_rsp.shape
            # If image to big then split into 3 parts along z axis. Also make sure that z-axis is at least 200px otherwise
            # splitting along it does not really make sense.
            do_triple_split = np.prod(ss) > nr_voxels_thr and ss[2] > 200 and multimodel
            if force_split:
                do_triple_split = True
            if do_triple_split:
                if not quiet: print("Splitting into subparts...")
                img_parts = ["s01", "s02", "s03"]
                third = img_in_rsp.shape[2] // 3
                margin = 20  # set margin with fixed values to avoid rounding problem if using percentage of third
                img_in_rsp_data = img_in_rsp.get_fdata()
                nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, :third+margin], img_in_rsp.affine),
                        tmp_dir / "s01_0000.nii.gz")
                nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, third+1-margin:third*2+margin], img_in_rsp.affine),
                        tmp_dir / "s02_0000.nii.gz")
                nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, third*2+1-margin:], img_in_rsp.affine),
                        tmp_dir / "s03_0000.nii.gz")

            if task_name == "total" and resample is not None and resample[0] < 3.0:
                # overall speedup for 15mm model roughly 11% (GPU) and 100% (CPU)
                # overall speedup for  3mm model roughly  0% (GPU) and  10% (CPU)
                # (dice 0.001 worse on test set -> ok)
                # (for lung_trachea_bronchia somehow a lot lower dice)
                step_size = 0.8
            else:
                step_size = 0.5

            st = time.time()
            if multimodel:  # if running multiple models

                # only compute model parts containing the roi subset
                if roi_subset is not None:
                    part_names = []
                    new_task_id = []
                    for part_name, part_map in class_map_parts.items():
                        if any(organ in roi_subset for organ in part_map.values()):
                            # get taskid associated to model part_name
                            map_partname_to_taskid = {v:k for k,v in map_taskid_to_partname.items()}
                            new_task_id.append(map_partname_to_taskid[part_name])
                            part_names.append(part_name)
                    task_id = new_task_id
                    if save_probs:
                        selected_classes = class_map[task_name]
                        selected_classes_inv_new_map = dict(zip([v for v in selected_classes.values() if v in roi_subset],
                                                                [i for i in range(len(roi_subset))]))

                    if verbose:
                        print(f"Computing parts: {part_names} based on the provided roi_subset")

                if test == 0:
                    class_map_inv = {v: k for k, v in class_map[task_name].items()}
                    (tmp_dir / "parts").mkdir(exist_ok=True)
                    seg_combined = {}
                    prob_combined = {}
                    # iterate over subparts of image
                    for img_part in img_parts:
                        img_shape = nib.load(tmp_dir / f"{img_part}_0000.nii.gz").shape
                        if not save_probs:
                            seg_combined[img_part] = np.zeros(img_shape, dtype=np.uint8)
                    # Run several tasks and combine results into one segmentation
                    if save_probs:
                        if roi_subset is not None:
                            num_classes = len(roi_subset)
                            prob_combined[img_part] = np.zeros(img_shape + (num_classes,), dtype=np.float32)
                        else:
                            for tmp_tid in task_id:
                                num_classes = num_classes + len(class_map_parts[map_taskid_to_partname[tmp_tid]])
                            prob_combined[img_part] = np.zeros(img_shape + (num_classes + 1,), dtype=np.float32)
                    for idx, tid in enumerate(task_id):
                        if not quiet: print(f"Predicting part {idx+1} of {len(task_id)} ...")
                        with nostdout(verbose):
                            # nnUNet_predict(tmp_dir, tmp_dir, tid, model, folds, trainer, tta,
                            #                nr_threads_resampling, nr_threads_saving)
                            nnUNetv2_predict(tmp_dir, tmp_dir, tid, model, folds, trainer, tta,
                                             nr_threads_resampling, nr_threads_saving,
                                             device=device, quiet=quiet, step_size=step_size, save_probs=save_probs)
                        # iterate over models (different sets of classes)
                        for img_part in img_parts:
                            if save_probs:
                                (tmp_dir / f"{img_part}.npz").rename(tmp_dir / "parts" / f"{img_part}_{tid}.npz")
                                prob = np.moveaxis(np.load(tmp_dir / "parts" / f"{img_part}_{tid}.npz")["probabilities"],
                                               [0, 1, 2, 3], [3, 2, 1, 0])
                            else:
                                (tmp_dir / f"{img_part}.nii.gz").rename(tmp_dir / "parts" / f"{img_part}_{tid}.nii.gz")
                                seg = nib.load(tmp_dir / "parts" / f"{img_part}_{tid}.nii.gz").get_fdata()
                            if save_probs:
                                if roi_subset is None:
                                    for jdx, class_name in class_map_parts[map_taskid_to_partname[tid]].items():
                                        prob_combined[img_part][:, :, :, class_map_inv[class_name]] = prob[:, :, :, jdx]
                                else:
                                    class_map_parts_sub = {k:v for k,v in class_map_parts[map_taskid_to_partname[tid]].items() if v in roi_subset}
                                    for jdx, class_name in class_map_parts_sub.items():
                                        prob_combined[img_part][:, :, :, selected_classes_inv_new_map[class_name]] = prob[:, :, :, jdx]
                            else:
                                for jdx, class_name in class_map_parts[map_taskid_to_partname[tid]].items():
                                        seg_combined[img_part][seg == jdx] = class_map_inv[class_name]
                        if save_probs:
                            # These files have to be deleted for nnunet to predict correctly
                            os.remove((tmp_dir / f"{img_part}.pkl"))
                            os.remove((tmp_dir / f"{img_part}.nii.gz"))
                    # iterate over subparts of image
                    for img_part in img_parts:
                        if save_probs:
                            prob_pred = nib.Nifti1Image(prob_combined[img_part], img_in_rsp.affine)
                        else:
                            nib.save(nib.Nifti1Image(seg_combined[img_part], img_in_rsp.affine),
                                     tmp_dir / f"{img_part}.nii.gz")
                elif test == 1:
                    print("WARNING: Using reference seg instead of prediction for testing.")
                    shutil.copy(Path("tests") / "reference_files" / "example_seg.nii.gz", tmp_dir / "s01.nii.gz")
            else:
                if not quiet: print("Predicting...")
                if test == 0:
                    with nostdout(verbose):
                        if save_probs:
                            num_classes = len(class_map_parts[map_taskid_to_partname[task_id]])
                        # nnUNet_predict(tmp_dir, tmp_dir, task_id, model, folds, trainer, tta,
                        #                nr_threads_resampling, nr_threads_saving)
                        nnUNetv2_predict(tmp_dir, tmp_dir, task_id, model, folds, trainer, tta,
                                         nr_threads_resampling, nr_threads_saving,
                                         device=device, quiet=quiet, step_size=step_size, save_probs=save_probs)
                        if save_probs:
                            prob_pred = nib.Nifti1Image(np.moveaxis(np.load(tmp_dir / f"{img_parts[0]}.npz")["probabilities"], [0,1,2,3], [3,2,1,0]), img_in_rsp.affine)

                # elif test == 2:
                #     print("WARNING: Using reference seg instead of prediction for testing.")
                #     shutil.copy(Path("tests") / "reference_files" / "example_seg_fast.nii.gz", tmp_dir / f"s01.nii.gz")
                elif test == 3:
                    print("WARNING: Using reference seg instead of prediction for testing.")
                    shutil.copy(Path("tests") / "reference_files" / "example_seg_lung_vessels.nii.gz", tmp_dir / "s01.nii.gz")
            if not quiet: print(f"  Predicted in {time.time() - st:.2f}s")

            # Combine image subparts back to one image (need to do this for augment and save_probs)
            if do_triple_split:
                combined_img = np.zeros(img_in_rsp.shape, dtype=np.uint8)
                combined_img[:,:,:third] = nib.load(tmp_dir / "s01.nii.gz").get_fdata()[:,:,:-margin]
                combined_img[:,:,third:third*2] = nib.load(tmp_dir / "s02.nii.gz").get_fdata()[:,:,margin-1:-margin]
                combined_img[:,:,third*2:] = nib.load(tmp_dir / "s03.nii.gz").get_fdata()[:,:,margin-1:]
                nib.save(nib.Nifti1Image(combined_img, img_in_rsp.affine), tmp_dir / "s01.nii.gz")

            if save_probs:
                if resample is not None:
                    if not quiet: print("Resampling...")
                    if verbose: print(f"  back to original shape: {img_in_shape}")
                    # Use force_affine otherwise output affine sometimes slightly off (which then is even increased
                    # by undo_canonical)
                    prob_pred = change_spacing(prob_pred, resample, img_in_shape + (num_classes,),
                                               order=0, dtype=np.float32, nr_cpus=6,
                                               force_affine=img_in.affine)

                if crop is not None:
                    prob_pred = undo_crop_probabilities(prob_pred, img_in_orig, bbox)

                prob_data = prob_pred.get_fdata().astype(np.float32)
                prob_header = img_in_orig.header.copy()
                prob_header.set_data_dtype(np.float32)
                prob_out = nib.Nifti1Image(prob_data, prob_pred.affine, prob_header)

                if file_out is not None and skip_saving is False:
                    for k, v in selected_classes_inv_new_map.items():
                        if not quiet:
                            print(f"Creating {k}_prob.nii.gz")
                        output_path = str(file_out / f"{k}_prob.nii.gz")
                        nib.save(nib.Nifti1Image(prob_data[:, :, :, v], prob_pred.affine, prob_header),
                                 output_path)

                return prob_out, selected_classes_inv_new_map
            else:
                img_pred = nib.load(tmp_dir / "s01.nii.gz")

                # Currently only relevant for T304 (appendicular bones)
                img_pred = remove_auxiliary_labels(img_pred, task_name)

                # Postprocessing multilabel (run here on lower resolution)
                if task_name == "body":
                    img_pred_pp = keep_largest_blob_multilabel(img_pred.get_fdata().astype(np.uint8),
                                                               class_map[task_name], ["body_trunc"], debug=False, quiet=quiet)
                    img_pred = nib.Nifti1Image(img_pred_pp, img_pred.affine)

                if task_name == "body":
                    vox_vol = np.prod(img_pred.header.get_zooms())
                    size_thr_mm3 = 50000
                    img_pred_pp = remove_small_blobs_multilabel(img_pred.get_fdata().astype(np.uint8),
                                                                class_map[task_name], ["body_extremities"],
                                                                interval=[size_thr_mm3/vox_vol, 1e10], debug=False, quiet=quiet)
                    img_pred = nib.Nifti1Image(img_pred_pp, img_pred.affine)

                # General postprocessing
                if remove_small_blobs:
                    if not quiet: print("Removing small blobs...")
                    st = time.time()
                    vox_vol = np.prod(img_pred.header.get_zooms())
                    size_thr_mm3 = 200
                    img_pred_pp = remove_small_blobs_multilabel(img_pred.get_fdata().astype(np.uint8),
                                                                class_map[task_name], list(class_map[task_name].values()),
                                                                interval=[size_thr_mm3/vox_vol, 1e10], debug=False, quiet=quiet)  # ~24s
                    img_pred = nib.Nifti1Image(img_pred_pp, img_pred.affine)
                    if not quiet: print(f"  Removed in {time.time() - st:.2f}s")

                if preview:
                    from totalsegmentator.preview import generate_preview
                    # Generate preview before upsampling so it is faster and still in canonical space
                    # for better orientation.
                    if not quiet: print("Generating preview...")
                    st = time.time()
                    smoothing = 20
                    preview_dir = file_out.parent if multilabel_image else file_out
                    generate_preview(img_in_rsp, preview_dir / f"preview_{task_name}.png", img_pred.get_fdata(), smoothing, task_name)
                    if not quiet: print(f"  Generated in {time.time() - st:.2f}s")

                # Statistics calculated on the 3mm downsampled image are very similar to statistics
                # calculated on the original image. Volume often completely identical. For intensity
                # some more change but still minor.
                #
                # Speed:
                # stats on 1.5mm: 37s
                # stats on 3.0mm: 4s    -> great improvement
                stats = None
                if statistics:
                    if not quiet: print("Calculating statistics fast...")
                    st = time.time()
                    if file_out is not None:
                        stats_dir = file_out.parent if multilabel_image else file_out
                        stats_dir.mkdir(exist_ok=True)
                        stats_file = stats_dir / "statistics.json"
                    else:
                        stats_file = None
                        stats = get_basic_statistics(img_pred.get_fdata(), img_in_rsp, stats_file,
                                                     quiet, task_name, exclude_masks_at_border, roi_subset,
                                                     metric=stats_aggregation)
                    if not quiet: print(f"  calculated in {time.time()-st:.2f}s")

                if resample is not None:
                    if not quiet: print("Resampling...")
                    if verbose: print(f"  back to original shape: {img_in_shape}")
                    # Use force_affine otherwise output affine sometimes slightly off (which then is even increased
                    # by undo_canonical)
                    img_pred = change_spacing(img_pred, resample, img_in_shape,
                                              order=0, dtype=np.uint8, nr_cpus=nr_threads_resampling,
                                              force_affine=img_in.affine)

                if verbose: print("Undoing canonical...")
                img_pred = undo_canonical(img_pred, img_in_orig)

                if crop is not None:
                    if verbose: print("Undoing cropping...")
                    img_pred = undo_crop(img_pred, img_in_orig, bbox)

                check_if_shape_and_affine_identical(img_in_orig, img_pred)
                img_data = img_pred.get_fdata().astype(np.uint8)
                if save_binary:
                    img_data = (img_data > 0).astype(np.uint8)

                # Reorder labels if needed
                if v1_order and task_name == "total":
                    img_data = reorder_multilabel_like_v1(img_data, class_map["total"], class_map["total_v1"])

                # Keep only voxel values corresponding to the roi_subset
                if roi_subset is not None:
                    img_data *= np.isin(img_data, list(label_map.keys()))

                # Prepare output nifti
                # Copy header to make output header exactly the same as input. But change dtype otherwise it will be
                # float or int and therefore the masks will need a lot more space.
                # (infos on header: https://nipy.org/nibabel/nifti_images.html)
                new_header = img_in_orig.header.copy()
                new_header.set_data_dtype(np.uint8)
                img_out = nib.Nifti1Image(img_data, img_pred.affine, new_header)
                img_out = add_label_map_to_nifti(img_out, label_map)

                if file_out is not None and skip_saving is False:
                    if not quiet: print("Saving segmentations...")

                    # Select subset of classes if required
                    selected_classes = class_map[task_name]
                    if roi_subset is not None:
                        selected_classes = {k:v for k, v in selected_classes.items() if v in roi_subset}

                    if output_type == "dicom":
                        file_out.mkdir(exist_ok=True, parents=True)
                        save_mask_as_rtstruct(img_data, selected_classes, file_in_dcm, file_out / "segmentations.dcm")
                    else:
                        st = time.time()
                        if multilabel_image:
                            file_out.parent.mkdir(exist_ok=True, parents=True)
                        else:
                            file_out.mkdir(exist_ok=True, parents=True)
                        if multilabel_image:
                            nib.save(img_out, file_out)
                            if nora_tag != "None":
                                subprocess.call(f"/opt/nora/src/node/nora -p {nora_tag} --add {file_out} --addtag atlas", shell=True)
                        else:  # save each class as a separate binary image
                            file_out.mkdir(exist_ok=True, parents=True)

                            if np.prod(img_data.shape) > 512*512*1000:
                                print("Shape of output image is very big. Setting nr_threads_saving=1 to save memory.")
                                nr_threads_saving = 1

                            # Code for single threaded execution  (runtime:24s)
                            if nr_threads_saving == 1:
                                for k, v in selected_classes.items():
                                    binary_img = img_data == k
                                    output_path = str(file_out / f"{v}.nii.gz")
                                    nib.save(nib.Nifti1Image(binary_img.astype(np.uint8), img_pred.affine, new_header), output_path)
                                    if nora_tag != "None":
                                        subprocess.call(f"/opt/nora/src/node/nora -p {nora_tag} --add {output_path} --addtag mask", shell=True)
                            else:
                                # Code for multithreaded execution
                                #   Speed with different number of threads:
                                #   1: 46s, 2: 24s, 6: 11s, 10: 8s, 14: 8s
                                nib.save(img_pred, tmp_dir / "s01.nii.gz")
                                _ = p_map(partial(save_segmentation_nifti, tmp_dir=tmp_dir, file_out=file_out, nora_tag=nora_tag, header=new_header, task_name=task_name, quiet=quiet),
                                        selected_classes.items(), num_cpus=nr_threads_saving, disable=quiet)

                                # Multihreaded saving with same functions as in nnUNet -> same speed as p_map
                                # pool = Pool(nr_threads_saving)
                                # results = []
                                # for k, v in selected_classes.items():
                                #     results.append(pool.starmap_async(save_segmentation_nifti, ((k, v, tmp_dir, file_out, nora_tag),) ))
                                # _ = [i.get() for i in results]  # this actually starts the execution of the async functions
                                # pool.close()
                                # pool.join()
                    if not quiet: print(f"  Saved in {time.time() - st:.2f}s")

                    # Postprocessing single files
                    #    (these not directly transferable to multilabel)

                    # Lung mask does not exist since I use 6mm model. Would have to save lung mask from 6mm seg.
                    # if task_name == "lung_vessels":
                    #     remove_outside_of_mask(file_out / "lung_vessels.nii.gz", file_out / "lung.nii.gz")

                    # if task_name == "heartchambers_test":
                    #     remove_outside_of_mask(file_out / "heart_myocardium.nii.gz", file_out / "heart.nii.gz", addon=5)
                    #     remove_outside_of_mask(file_out / "heart_atrium_left.nii.gz", file_out / "heart.nii.gz", addon=5)
                    #     remove_outside_of_mask(file_out / "heart_ventricle_left.nii.gz", file_out / "heart.nii.gz", addon=5)
                    #     remove_outside_of_mask(file_out / "heart_atrium_right.nii.gz", file_out / "heart.nii.gz", addon=5)
                    #     remove_outside_of_mask(file_out / "heart_ventricle_right.nii.gz", file_out / "heart.nii.gz", addon=5)
                    #     remove_outside_of_mask(file_out / "aorta.nii.gz", file_out / "heart.nii.gz", addon=5)
                    #     remove_outside_of_mask(file_out / "pulmonary_artery.nii.gz", file_out / "heart.nii.gz", addon=5)

                    if task_name == "body" and not multilabel_image and not no_derived_masks:
                        if not quiet: print("Creating body.nii.gz")
                        body_img = combine_masks(file_out, "body")
                        nib.save(body_img, file_out / "body.nii.gz")
                        if not quiet: print("Creating skin.nii.gz")
                        skin = extract_skin(img_in_orig, nib.load(file_out / "body.nii.gz"))
                        nib.save(skin, file_out / "skin.nii.gz")
            return img_out, img_in_orig, stats

