from totalsegmentator.python_api import totalsegmentator
import nibabel as nib

if __name__ == '__main__':

    abdomen_subset = ['spleen',
                      'kidney_left',
                      'kidney_right',
                      'gallbladder',
                      'liver',
                      'stomach',
                      'pancreas',
                      'adrenal_gland_left',
                      'adrenal_gland_right',
                      'small_bowel',
                      'duodenum',
                      'colon',
                      'urinary_bladder',
                      'prostate']

    ####################################################################################################################
    #
    # Option 1: provide input and output paths
    #
    #   totalsegmentator(in_path, out_path, roi_subset=abdomen_subset)
    #
    #   totalsegmentator(in_path, out_path, save_probs=True, roi_subset=abdomen_subset)
    #
    #   totalsegmentator(in_path, out_path, augment=True, roi_subset=abdomen_subset)
    #
    ####################################################################################################################
    #
    # Option 2: provide input Nifti image
    #
    #   seg = totalsegmentator(img, roi_subset=abdomen_subset)
    #
    #   prob, classes = totalsegmentator(img, save_probs=True, roi_subset=abdomen_subset)
    #
    #   seg, unc, classes = totalsegmentator(img, augment=True, roi_subset=abdomen_subset)
    #
    ####################################################################################################################

    img = nib.load("test.nii.gz")

    # Test time augmentation
    seg, unc, classes = totalsegmentator(img, augment=True, roi_subset=abdomen_subset)