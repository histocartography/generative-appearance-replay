import os
import numpy as np
import pandas as pd
import torch
from PIL import Image

from torch.utils import data
from torchvision import transforms


def augment_data(image, annotation, tissue_mask):
    """Augment image and mask

    Args:
        image: image
        annotation: ground truth segmentation mask
        tissue_mask: pure background mask

    Returns:
        Flipped and/or rotated image and masks.
    """
    image = np.array(image)
    annotation = np.array(annotation)
    tissue_mask = np.array(tissue_mask)

    if np.random.rand() > 0.5:
        ax = np.random.choice([0, 1])
        image = np.flip(image, axis=ax)
        annotation = np.flip(annotation, axis=ax)
        tissue_mask = np.flip(tissue_mask, axis=ax)

    if np.random.rand() > 0.5:
        rot = np.random.choice([1, 2, 3])  # 90, 180, or 270 degrees
        image = np.rot90(image, k=rot, axes=[0, 1])
        annotation = np.rot90(annotation, k=rot, axes=[0, 1])
        tissue_mask = np.rot90(tissue_mask, k=rot, axes=[0, 1])

    return image, annotation, tissue_mask


class BaseDataset(data.Dataset):
    """Base dataset class."""

    def __init__(
        self,
        csv_path,
        mode,
        split_path,
        train_gan=False,
        dataset=None,
        split=None,
        info_path=None,
    ):
        """Initialize base dataset class.

        Args:
            csv_path: path to all csv file with all image names
            mode: train/val/test
            split_path: path to train/val/test split file (csv)
            train_gan: whether to do GAN training. Defaults to False.
            dataset: dataset name. Defaults to None.
            split: split name (e.g., of a scanning device). Defaults to None.
            info_path: path to file with additional info. Defaults to None.
        """
        self.mode = mode
        self.train_gan = train_gan
        self.dataset = dataset.lower() if dataset is not None else None
        self.split = split.lower() if split is not None else None
        self.info = None

        if not self.train_gan:
            split = pd.read_csv(split_path)[self.mode]
            split = split.dropna().reset_index(drop=True)
            assert len(split) > 0, "Split should not be empty"

            # prepare slide data
            slide_data = pd.read_csv(csv_path)
            mask = slide_data["slide_id"].isin(split.tolist())
            self.slide_data = slide_data[mask].reset_index(drop=True)
            if info_path is not None:
                self.info = pd.read_csv(info_path)
        else:
            train_split = pd.read_csv(split_path)["train"].dropna().reset_index(drop=True)
            val_split = pd.read_csv(split_path)["val"].dropna().reset_index(drop=True)
            slide_data = pd.read_csv(csv_path)
            mask = slide_data["slide_id"].isin(train_split.tolist() + val_split.tolist())
            self.slide_data = slide_data[mask].reset_index(drop=True)

    def __len__(self):
        return None

    def __getitem__(self, idx):
        return None


class ODSDataset(BaseDataset):
    """Dataset class for optic disc segmentation (ODS)."""

    def __init__(
        self,
        image_path,
        anno_path,
        mask_path,
        test=False,
        augmentation=False,
        **kwargs,
    ):
        """Initialize ODS dataset class.

        Args:
            image_path: path to images
            anno_path: path to ground truth segmentation masks
            mask_path: path to background masks
            test: test mode. Defaults to False.
            augmentation: whether to apply augmentations. Defaults to False.
        """
        super(ODSDataset, self).__init__(**kwargs)
        self.image_path = image_path
        self.anno_path = anno_path
        self.mask_path = mask_path
        self.test = test
        self.augmentation = augmentation

        self.image_names = [image_name + ".png" for image_name in self.slide_data["slide_id"].tolist()]
        print(f"Total number of {self.mode} images: {len(self.image_names)}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        image = Image.open(os.path.join(self.image_path, image_name))
        annotation = Image.open(os.path.join(self.anno_path, image_name))
        tissue_mask = Image.open(os.path.join(self.mask_path, image_name))

        if self.augmentation:
            image, annotation, tissue_mask = augment_data(image, annotation, tissue_mask)

        if not self.train_gan:
            annotation = torch.from_numpy(np.array(annotation)).type(torch.LongTensor)
            return (transforms.ToTensor()(image.copy()), annotation, np.array(tissue_mask))
        else:
            return transforms.ToTensor()(image.copy())


class MnMsDataset(BaseDataset):
    """MnMs dataset class for cardiac segmentation (CS)."""

    def __init__(self, image_path, anno_path, test=False, augmentation=False, **kwargs):
        """Initialize CS dataset class.

        Args:
            image_path: path to images
            anno_path: path to ground truth segmentation masks
            test: test mode. Defaults to False.
            augmentation: whether to apply augmentations. Defaults to False.
        """

        super(MnMsDataset, self).__init__(**kwargs)
        self.image_path = image_path
        self.anno_path = anno_path
        self.test = test
        self.augmentation = augmentation

        if self.mode == "test":
            pat2slides = {}
            for s_id in self.slide_data["slide_id"].tolist():
                subject_id = s_id.split("-")[0].replace("subject", "")
                frame_id = int(s_id.split("_")[-2].replace("frame", ""))
                ed_frame_id = int(self.info[self.info["External code"] == subject_id]["ED"])
                es_frame_id = int(self.info[self.info["External code"] == subject_id]["ES"])
                if subject_id not in pat2slides.keys():
                    if frame_id == ed_frame_id:
                        pat2slides[subject_id] = {"ED": [s_id]}
                    elif frame_id == es_frame_id:
                        pat2slides[subject_id] = {"ES": [s_id]}
                    else:
                        raise ValueError(f"Slice {s_id} cannot be mapped to ED/ES")
                else:
                    if frame_id == ed_frame_id:
                        if "ED" not in pat2slides[subject_id].keys():
                            pat2slides[subject_id]["ED"] = [s_id]
                        else:
                            pat2slides[subject_id]["ED"].append(s_id)
                    elif frame_id == es_frame_id:
                        if "ES" not in pat2slides[subject_id].keys():
                            pat2slides[subject_id]["ES"] = [s_id]
                        else:
                            pat2slides[subject_id]["ES"].append(s_id)
                    else:
                        raise ValueError(f"Slice {s_id} cannot be mapped to ED/ES")
            self.image_names = [v for v in pat2slides.values()]
            print(f"Total number of {self.mode} images: {len(self.image_names)}")
        else:
            self.image_names = [image_name for image_name in self.slide_data["slide_id"].tolist()]
            print(f"Total number of {self.mode} images: {len(self.image_names)}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "test":
            patient_ed = []
            patient_es = []
            for phase in self.image_names[idx].keys():
                for image_name in self.image_names[idx][phase]:
                    image = Image.open(os.path.join(self.image_path, f"{image_name}.png"))
                    annotation = torch.from_numpy(np.load(os.path.join(self.anno_path, f"{image_name}.npy"))).type(
                        torch.LongTensor
                    )
                    tissue_mask = np.ones(image.size)
                    if phase == "ED":
                        patient_ed.append((transforms.ToTensor()(image.copy()), annotation, np.ones(image.size)))
                    else:
                        patient_es.append((transforms.ToTensor()(image.copy()), annotation, np.ones(image.size)))
            return patient_ed, patient_es

        else:
            image_name = self.image_names[idx]

            image = Image.open(os.path.join(self.image_path, f"{image_name}.png"))
            annotation = np.load(os.path.join(self.anno_path, f"{image_name}.npy"))
            tissue_mask = np.ones(image.size)

            if self.augmentation:
                image, annotation, tissue_mask = augment_data(image, annotation, tissue_mask)

            if not self.train_gan:
                annotation = torch.from_numpy(np.array(annotation)).type(torch.LongTensor)
                return transforms.ToTensor()(image.copy()), annotation, tissue_mask
            else:
                return transforms.ToTensor()(image.copy())


class MSProstateDataset(BaseDataset):
    """Multi-site (MS) dataset class for prostate segmentation (PS)."""

    def __init__(
        self,
        image_path,
        anno_path,
        test=False,
        augmentation=False,
        **kwargs,
    ):
        """Initialize PS dataset class.

        Args:
            image_path: path to images
            anno_path: path to ground truth segmentation masks
            test: test mode. Defaults to False.
            augmentation: whether to apply augmentations. Defaults to False.
        """
        super(MSProstateDataset, self).__init__(**kwargs)
        self.image_path = image_path
        self.anno_path = anno_path
        self.test = test
        self.augmentation = augmentation

        if self.mode == "test":
            pat2slides = {}
            for s_id in self.slide_data["slide_id"].tolist():
                case_id = s_id.split("_slice")[0]
                if case_id not in pat2slides.keys():
                    pat2slides[case_id] = [s_id]
                else:
                    pat2slides[case_id].append(s_id)
            self.image_names = [v for v in pat2slides.values()]
            print(f"Total number of {self.mode} images: {len(self.image_names)}")
        else:
            self.image_names = [image_name for image_name in self.slide_data["slide_id"].tolist()]
            print(f"Total number of {self.mode} images: {len(self.image_names)}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        if self.mode == "test":
            patient = []
            for image_name in self.image_names[idx]:
                image = Image.open(os.path.join(self.image_path, f"{image_name}.png"))
                annotation = torch.from_numpy(np.load(os.path.join(self.anno_path, f"{image_name}.npy"))).type(
                    torch.LongTensor
                )
                tissue_mask = np.ones(image.size)
                patient.append((transforms.ToTensor()(image.copy()), annotation, np.ones(image.size)))
            return patient
        else:
            image_name = self.image_names[idx]

            image = Image.open(os.path.join(self.image_path, f"{image_name}.png"))
            annotation = np.load(os.path.join(self.anno_path, f"{image_name}.npy"))
            tissue_mask = np.ones(image.size)

            if self.augmentation:
                image, annotation, tissue_mask = augment_data(image, annotation, tissue_mask)

            if not self.train_gan:
                annotation = torch.from_numpy(np.array(annotation)).type(torch.LongTensor)
                return transforms.ToTensor()(image.copy()), annotation, tissue_mask
            else:
                return transforms.ToTensor()(image.copy())
