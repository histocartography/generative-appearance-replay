import os
import glob
from pathlib import Path

import numpy as np
from PIL import Image
import SimpleITK as sitk
import pandas as pd


class MSProstatePreprocessor:
    """Preprocessing for the multi-site (MS) prostate segmentation dataset.

    Adapted from https://github.com/liuquande/MS-Net.
    """

    def __init__(self, base_in, base_out):
        """Initialize the preprocessor.

        Args:
            base_in: base input path (raw data)
            base_out: base output path (preprocessed data)
        """
        self.base_in = base_in
        self.base_out = base_out

    def extract_patch(self, filenames, patch_size):
        """Extract image patch around mask area.

        Args:
            filenames: tuple of image and mask path
            patch_size: image patch size

        Returns:
            image and mask patch
        """
        image, mask = self.parse_fn(filenames)

        _, _, lim_z = np.where(mask > 0)
        min_z, max_z = np.min(lim_z), np.max(lim_z)
        min_x = int(np.ceil((image.shape[0] - patch_size) / 2))
        max_x = image.shape[0] - min_x
        min_y = int(np.ceil((image.shape[1] - patch_size) / 2))
        max_y = image.shape[1] - min_y
        image_patch = image[min_x:max_x, min_y:max_y, min_z : max_z + 1]
        mask_patch = mask[min_x:max_x, min_y:max_y, min_z : max_z + 1]

        return image_patch.astype(np.float32), mask_patch.astype(np.float32)

    def parse_fn(self, data_path):
        """Load image and mask as numpy arrays and normalize.

        Args:
            data_path: tuple of image and mask path

        Returns:
            normalized image and mask arrays
        """
        # load image and mask
        image_path = data_path[0]
        label_path = data_path[1]
        itk_image = sitk.ReadImage(image_path)
        itk_mask = sitk.ReadImage(label_path)
        image = sitk.GetArrayFromImage(itk_image)
        mask = sitk.GetArrayFromImage(itk_mask)

        # normalize per image using statistics within the prostate, but apply to the whole image
        binary_mask = np.ones(mask.shape)
        mean = np.sum(image * binary_mask) / np.sum(binary_mask)
        std = np.sqrt(np.sum(np.square(image - mean) * binary_mask) / np.sum(binary_mask))
        image = (image - mean) / std

        # set label
        mask[mask == 2] = 1

        # transpose the orientation
        return image.transpose([1, 2, 0]), mask.transpose([1, 2, 0])

    def run(self):
        """Run preprocessing for all sites."""

        print(
            "WARNING: the preprocessing expects the directory names: "
            + "Site_A (RUNMC), Site_B (BMC), Site_C (I2CVB), Site_D (UCL), Site_E (BIDMC), Site_F (HK)."
        )
        for site in ["Site_A", "Site_B", "Site_C", "Site_D", "Site_E", "Site_F"]:
            all_slides = []
            if site == "Site_B":
                mask_paths = sorted(glob.glob(f"{self.base_in}/{site}/*Segmentation.nii.gz"))
            else:
                mask_paths = sorted(glob.glob(f"{self.base_in}/{site}/*segmentation.nii.gz"))
            os.makedirs(f"{self.base_out}/{site}/images", exist_ok=True)
            os.makedirs(f"{self.base_out}/{site}/masks", exist_ok=True)
            print(f"Preprocessing {len(mask_paths)} images of {site}...")
            for mpath in mask_paths:
                ipath = "".join(mpath.split("_Segmentation" if site == "Site_B" else "_segmentation"))
                fname = Path(ipath).name.split(".nii.gz")[0]
                img, mask = self.extract_patch(filenames=(ipath, mpath), patch_size=256)
                for slice_idx in range(img.shape[-1]):
                    s_id = "%02d" % slice_idx
                    img_slice = img[..., slice_idx]
                    mask_slice = mask[..., slice_idx]
                    # save image slice
                    img_slice = Image.fromarray(
                        np.uint8(255 * (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())), "L"
                    )
                    img_slice.save(f"{self.base_out}/{site}/images/{fname}_{site}_slice{s_id}.png")
                    # save mask slice
                    np.save(f"{self.base_out}/{site}/masks/{fname}_{site}_slice{s_id}.npy", mask_slice)
                    all_slides.append(f"{fname}_{site}_slice{s_id}")

            # save all_slides.csv
            pd.DataFrame({"slide_id": all_slides}).to_csv(f"{self.base_out}/{site}/all_slides.csv")

            # save split.csv
            self.create_split(slide_ids=all_slides, out_path=f"{self.base_out}/{site}")

    def create_split(self, slide_ids, out_path, train_ratio=0.6, val_ratio=0.15):
        """Create .csv file for train/val/test split.

        Args:
            slide_ids: list of all slide (image) IDs
            out_path: output path
            train_ratio: ratio of training data. Defaults to 0.6.
            val_ratio: ratio of validation data. Defaults to 0.15.
        """
        # create patient dictionary
        patient2slides = {}
        for s_id in slide_ids:
            subject_id = s_id.split("_")[0]
            if subject_id not in patient2slides.keys():
                patient2slides[subject_id] = [s_id]
            else:
                patient2slides[subject_id].append(s_id)

        # shuffle patient IDs
        patient_ids = sorted(patient2slides.keys())
        indices = list(range(len(patient_ids)))
        np.random.shuffle(indices)
        patient_ids = [patient_ids[i] for i in indices]

        # split IDs in train/val/test
        n_train = int(train_ratio * len(patient_ids))
        n_val = int(val_ratio * len(patient_ids))
        train_p_ids = patient_ids[:n_train]
        val_p_ids = patient_ids[n_train : n_train + n_val]
        test_p_ids = patient_ids[n_train + n_val :]

        # create split dictionary
        split_dict = {
            "train": pd.Series([s_id for p_id in train_p_ids for s_id in patient2slides[p_id]]),
            "val": pd.Series([s_id for p_id in val_p_ids for s_id in patient2slides[p_id]]),
            "test": pd.Series([s_id for p_id in test_p_ids for s_id in patient2slides[p_id]]),
        }

        # save as .csv
        df = pd.DataFrame(split_dict)
        df.to_csv(f"{out_path}/split.csv")
