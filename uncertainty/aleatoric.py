from scipy.ndimage import zoom
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import numpy as np
import nibabel as nib
import random

class test_time_augmentation():
    def __init__(self, img):
        self.img = img
        self.img_data = img.get_fdata()
        self.affine = img.affine
        self.rot_angle = None
        self.mirror_axis = None

    def do_augment(self, do_rotation=False, do_gauss_noise=False, do_gauss_blur=False,
                   do_brightness=False, do_contrast=False, do_low_res=False, do_gamma=False,
                   do_mirror=False):
        def rotation(self, img, affine):

            # The definition of this angle changes depending on the patch size in the original nnunet paper
            angle = random.randint(-30, 30)

            # Rotation angle
            self.rot_angle = angle

            # Is there a way to do this just by altering the nifti affine matrix?
            img = rotate(img, angle=angle, mode="constant",cval=-1024, reshape=False)
            img = nib.Nifti1Image(img, affine)
            return img

        def gaussian_noise(img, affine):
            # Add zero mean gaussian noise to the normalized image
            variance = random.uniform(0, 0.1)
            mn = np.mean(img)
            sd = np.std(img)

            # Reverse normalization applied to noise image (Correct?)
            noise = (np.random.normal(0, variance, img.shape) * sd) + mn
            img = img + noise

            img = nib.Nifti1Image(img, affine)
            return img

        def gaussian_blur(img, affine):
            # Randomly sample filter standard deviation
            sigma = random.uniform(0.5, 1.5)

            # Filter image
            img = gaussian_filter(img, sigma=sigma)
            img = nib.Nifti1Image(img, affine)
            return img

        def brightness(img, affine):
            # Randomly select scale factor
            scale_factor = random.uniform(0.7, 1.3)

            # Scale brightness values
            img = img * scale_factor
            img = nib.Nifti1Image(img, affine)
            return img

        def contrast(img, affine):
            # Randomly sample scale factor
            scale_factor = random.uniform(0.65, 1.5)

            # Identify minumum and maximum values
            mx = np.max(img)
            mn = np.min(img)

            # Scale image
            img = img * scale_factor

            # Clip min and max values
            img[img > mx] = mx
            img[img < mn] = mn
            img = nib.Nifti1Image(img, affine)
            return img

        def low_res(img, affine):
            # Randomly sample scale factor
            down_sample_factor = random.uniform(1, 2)

            # Create new dims
            new_shape = tuple(np.round(x / down_sample_factor) for x in img.shape[:2]) + (img.shape[2],)

            # Downsample (nearest) and resample (cubic) image
            img_down = resize(img, new_shape, order=0)
            img_up = resize(img_down, img.shape, order=3)
            img = nib.Nifti1Image(img_up, affine)
            return img

        def gamma_augmentation(img, affine):
            # normalize image to [0, 1]
            mx = np.max(img)
            mn = np.min(img)
            img = (img - mn)/(mx - mn)

            # Sample the gamma parameter
            gamma = np.random.uniform(0.7, 1.5)

            if np.random.uniform(0, 1) < 0.85:
                # Scale intensities
                img = img ** gamma
            else:
                img = 1 - (1 - img) ** gamma

            # Rescale to original value range
            img = img * (mx - mn) + mn
            img = nib.Nifti1Image(img, affine)
            return img

        def mirroring(img, affine):
            self.mirror_axis = np.round(np.random.uniform(0, 1))
            if self.mirror_axis == 0:
                img = img[::-1, :, :]
            else:
                img = img[:, ::-1, :]
            img = nib.Nifti1Image(img, affine)
            return img

        aug = {'original': self.img}
        if do_rotation:
            aug['rotation'] = rotation(self, self.img_data, self.affine)
        if do_gauss_noise:
            aug['gauss_noise'] = gaussian_noise(self.img_data, self.affine)
        if do_gauss_blur:
            aug['gauss_blur'] = gaussian_blur(self.img_data, self.affine)
        if do_brightness:
            aug['brightness'] = brightness(self.img_data, self.affine)
        if do_contrast:
            aug['contrast'] = contrast(self.img_data, self.affine)
        if do_low_res:
            aug['low_res'] = low_res(self.img_data, self.affine)
        if do_gamma:
            aug['gamma'] = gamma_augmentation(self.img_data, self.affine)
        if do_mirror:
            aug['mirror'] = mirroring(self.img_data, self.affine)
        return aug

    def undo_augment(self, probs):
        def undo_rotation(img, affine):
            img = rotate(img, angle=-self.rot_angle, mode="constant", cval=0.0, reshape=False)
            img = nib.Nifti1Image(img, affine)
            return img

        def undo_mirror(img, affine):
            if self.mirror_axis == 0:
                img = img[::-1, :, :, :]
            else:
                img = img[:, ::-1, :, :]
            img = nib.Nifti1Image(img, affine)
            return img

        if 'rotation' in probs:
            img = probs['rotation'].get_fdata()
            affine = probs['rotation'].affine
            probs['rotation'] = undo_rotation(img, affine)
        if 'mirror' in probs:
            img = probs['mirror'].get_fdata()
            affine = probs['mirror'].affine
            probs['mirror'] = undo_mirror(img, affine)
        return probs

    def compute_entropy(self, probs):

        # Get data shape
        size = probs['original'].shape

        # Get number of augmentations
        num_augs = len(probs)

        # Initialize uncertainty array
        unc = np.zeros(size[:-1] + (num_augs,))
        eps = 1e-10

        seg = np.zeros(size[:-1] + (size[-1] + 1,) + (num_augs,), dtype=np.uint8)
        rng = np.zeros(size + (num_augs,), dtype=np.float32)
        for i, v in enumerate(probs.values()):
            v = v.get_fdata()
            s = v > 0.5
            seg[:, :, :, 1:, i] = s.astype(np.uint8)
            rng[:, :, :, :, i] = v

        # Create multilabel image
        seg = np.argmax(seg, axis=3).astype(np.uint8)
        seg = nib.Nifti1Image(seg, probs['original'].affine)

        entropy = True
        variance = False

        if entropy:
            for i, v in enumerate(probs.values()):
                ent = np.zeros(size, dtype=np.float32)
                class_prob = v.get_fdata()
                class_prob[class_prob < eps] = eps
                class_prob[class_prob > 1 - eps] = 1 - eps
                ent = np.maximum(ent, -(class_prob * np.log2(class_prob) + (1 - class_prob) * np.log2(1 - class_prob)))
                ent = np.max(ent, axis=3)
                unc[:, :, :, i] = ent

        if variance:
            unc = np.zeros(size, dtype=np.float32)
            for c in range(size[-1]):
                class_probs = np.zeros(size[:-1] + (num_augs,), dtype=np.float32)
                for i, v in enumerate(probs.values()):
                    class_probs[:,:,:,i] = v.get_fdata()[:,:,:,c]
                unc[:,:,:,c] = np.var(class_probs, axis=3)

        # Create Max intensity projection
        # unc = np.max(unc, axis=3)


        # Create Nifti Image
        unc = nib.Nifti1Image(unc, probs['original'].affine)

        return seg, unc