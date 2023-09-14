import os
import torch
import numpy as np


def get_device():
    """Get device (cpu or gpu)."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model, load_path=""):
    """Load model checkpoint."""
    model.load_state_dict(torch.load(load_path))
    return model


def save_checkpoint(model, save_path=""):
    """Save a checkpoint model."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path))


def mnms_collate_fn(batch):
    """Collate function for test samples from the MnMs dataset."""
    images_ed = [item[0] for item in batch[0]]
    images_es = [item[0] for item in batch[1]]
    annotations_ed = [item[1] for item in batch[0]]
    annotations_es = [item[1] for item in batch[1]]
    tissue_masks_ed = [item[2] for item in batch[0]]
    tissue_masks_es = [item[2] for item in batch[1]]
    images_ed = torch.stack(images_ed)
    images_es = torch.stack(images_es)
    annotations_ed = torch.stack(annotations_ed)
    annotations_es = torch.stack(annotations_es)
    tissue_masks_ed = np.stack(tissue_masks_ed)
    tissue_masks_es = np.stack(tissue_masks_es)
    return (images_ed, annotations_ed, tissue_masks_ed), (images_es, annotations_es, tissue_masks_es)


def ps_collate_fn(batch):
    """Collate function for test samples from the MS-Prostate dataset."""
    images = torch.stack([item[0] for item in batch])
    annotations = torch.stack([item[1] for item in batch])
    tissue_masks = np.stack([item[2] for item in batch])
    return (images, annotations, tissue_masks)
